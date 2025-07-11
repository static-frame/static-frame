from __future__ import annotations

import copy
import dataclasses
import datetime
import io
import itertools as it
import os
import pickle
import string
import unittest
from collections import OrderedDict, defaultdict, namedtuple
from functools import partial
from hashlib import sha256
from io import StringIO
from itertools import chain, repeat
from tempfile import TemporaryDirectory, TemporaryFile

import frame_fixtures as ff
import numpy as np
import pytest
import typing_extensions as tp

import static_frame as sf
from static_frame import (
    DisplayConfig,
    Frame,
    FrameGO,
    FrameHE,
    HLoc,
    ILoc,
    Index,
    IndexAutoConstructorFactory,
    IndexAutoFactory,
    IndexDate,
    IndexDateGO,
    IndexDefaultConstructorFactory,
    IndexGO,
    IndexHierarchy,
    IndexHierarchyGO,
    IndexSecond,
    IndexYear,
    IndexYearGO,
    IndexYearMonth,
    Series,
    TypeBlocks,
    isna_element,
    mloc,
)
from static_frame.core.exception import (
    AxisInvalid,
    ErrorInitColumns,
    ErrorInitFrame,
    ErrorInitIndex,
    ErrorNPYEncode,
    ImmutableTypeError,
    InvalidDatetime64Initializer,
    InvalidFillValue,
)
from static_frame.core.fill_value_auto import FillValueAuto
from static_frame.core.frame import FrameAssignBLoc, FrameAssignILoc
from static_frame.core.store_config import StoreConfig
from static_frame.core.store_filter import StoreFilter
from static_frame.core.store_xlsx import StoreXLSX
from static_frame.core.util import (
    STORE_LABEL_DEFAULT,
    TLabel,
    WarningsSilent,
    iloc_to_insertion_iloc,
)
from static_frame.test.test_case import TestCase, skip_win, temp_file

if tp.TYPE_CHECKING:
    import sqlite3


nan = np.nan


class TestUnit(TestCase):
    def test_frame_slotted_a(self) -> None:
        f1 = Frame.from_element(1, index=(1, 2), columns=(3, 4, 5))

        with self.assertRaises(AttributeError):
            f1.g = 30  # type: ignore
        with self.assertRaises(AttributeError):
            f1.__dict__

    def test_frame_init_a(self) -> None:
        f = Frame.from_dict(OrderedDict([('a', (1, 2)), ('b', (3, 4))]), index=('x', 'y'))
        self.assertEqual(
            f.to_pairs(), (('a', (('x', 1), ('y', 2))), ('b', (('x', 3), ('y', 4))))
        )

        f = Frame.from_dict(OrderedDict([('b', (3, 4)), ('a', (1, 2))]), index=('x', 'y'))
        self.assertEqual(
            f.to_pairs(), (('b', (('x', 3), ('y', 4))), ('a', (('x', 1), ('y', 2))))
        )

    def test_frame_init_b(self) -> None:
        # test unusual instantiation cases

        # create a frame with a single value
        f1 = Frame.from_element(1, index=(1, 2), columns=(3, 4, 5))
        self.assertEqual(
            f1.to_pairs(),
            ((3, ((1, 1), (2, 1))), (4, ((1, 1), (2, 1))), (5, ((1, 1), (2, 1)))),
        )

        # with columns not defined, we create a DF with just an index
        f2 = FrameGO(index=(1, 2))
        f2['a'] = (-1, -1)
        self.assertEqual(f2.to_pairs(), (('a', ((1, -1), (2, -1))),))

        # with columns and index defined, we fill the value even if None
        f3 = Frame.from_element(None, index=(1, 2), columns=(3, 4, 5))
        self.assertEqual(
            f3.to_pairs(),
            (
                (3, ((1, None), (2, None))),
                (4, ((1, None), (2, None))),
                (5, ((1, None), (2, None))),
            ),
        )

        # auto populated index/columns based on shape
        f4 = Frame.from_records([[1, 2], [3, 4], [5, 6]])
        self.assertEqual(
            f4.to_pairs(),
            ((0, ((0, 1), (1, 3), (2, 5))), (1, ((0, 2), (1, 4), (2, 6)))),
        )
        self.assertTrue(f4._index._map is None)  # type: ignore
        self.assertTrue(f4._columns._map is None)  # type: ignore

    def test_frame_init_c(self) -> None:
        f = sf.FrameGO.from_dict(dict(color=('black',)))
        s = f['color']
        self.assertEqual(s.to_pairs(), ((0, 'black'),))

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

        self.assertEqual(
            f.to_pairs(),
            (('a', ((0, 1), (1, 2), (2, 3))), ('b', ((0, 4), (1, 5), (2, 6)))),
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

        self.assertEqual(
            f1.to_pairs(),
            (
                ('x', (('a', 3), ('b', 4), ('c', 5))),
                ('y', (('a', 12), ('b', 10), ('c', 11))),
            ),
        )

    def test_frame_init_j(self) -> None:
        f1 = sf.Frame.from_element('q', index=tuple('ab'), columns=tuple('xy'))
        self.assertEqual(
            f1.to_pairs(),
            (('x', (('a', 'q'), ('b', 'q'))), ('y', (('a', 'q'), ('b', 'q')))),
        )

    def test_frame_init_k(self) -> None:
        # check that we got autoincrement indices if no col/index provided
        f1 = Frame.from_records([[0, 1], [2, 3]])
        self.assertEqual(f1.to_pairs(), ((0, ((0, 0), (1, 2))), (1, ((0, 1), (1, 3)))))

    def test_frame_init_m(self) -> None:
        # cannot create a single element filled Frame specifying a shape (with index and columns) but not specifying a data value
        with self.assertRaises(RuntimeError):
            f1 = Frame(index=(3, 4, 5), columns=list('abv'))

    def test_frame_init_n(self) -> None:
        f1 = Frame.from_element(None, index=(3, 4, 5), columns=())
        self.assertEqual(f1.shape, (3, 0))

    def test_frame_init_o(self) -> None:
        f1 = Frame()
        self.assertEqual(f1.shape, (0, 0))

    def test_frame_init_p(self) -> None:
        # raise when a data values ir provided but an axis is size zero

        f1 = sf.Frame.from_element('x', index=(1, 2, 3), columns=iter(()))
        self.assertEqual(f1.shape, (3, 0))

        f2 = sf.Frame.from_element(None, index=(1, 2, 3), columns=iter(()))
        self.assertEqual(f2.shape, (3, 0))

    def test_frame_init_q(self) -> None:
        f1 = sf.Frame(index=(1, 2, 3), columns=iter(()))
        self.assertEqual(f1.shape, (3, 0))
        self.assertEqual(f1.to_pairs(), ())

    def test_frame_init_r(self) -> None:
        f1 = sf.Frame(index=(), columns=iter(range(3)))

        self.assertEqual(f1.shape, (0, 3))
        self.assertEqual(f1.to_pairs(), ((0, ()), (1, ()), (2, ())))

        # can create an un fillable Frame when using from_element
        f1 = sf.Frame.from_element('x', index=(), columns=iter(range(3)))

    def test_frame_init_s(self) -> None:
        # check that we got autoincrement indices if no col/index provided
        f1 = Frame.from_records(
            [[0, 1], [2, 3]], index=IndexAutoFactory, columns=IndexAutoFactory
        )

        self.assertEqual(f1.to_pairs(), ((0, ((0, 0), (1, 2))), (1, ((0, 1), (1, 3)))))

        f2 = Frame.from_records(
            [[0, 1], [2, 3]], index=IndexAutoFactory, columns=list('ab')
        )
        self.assertEqual(
            f2.to_pairs(), (('a', ((0, 0), (1, 2))), ('b', ((0, 1), (1, 3))))
        )

    def test_frame_init_t(self) -> None:
        # 3d array raises exception
        a1 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

        with self.assertRaises(RuntimeError):
            f1 = Frame(a1)

    def test_frame_init_u1(self) -> None:
        # 3d array raises exception
        a1 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        with self.assertRaises(RuntimeError):
            f1 = Frame(a1)

    def test_frame_init_u2(self) -> None:
        # NOTE: presently the inner lists get flattend when used in from records
        a1 = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
        f1 = Frame.from_records(a1)

        self.assertEqual(
            f1.to_pairs(),
            ((0, ((0, [1, 2]), (1, [5, 6]))), (1, ((0, [3, 4]), (1, [7, 8])))),
        )

    def test_frame_init_v(self) -> None:
        s1 = Series(['a', 'b', 'c'])

        with self.assertRaises(ErrorInitFrame):
            f1 = Frame(s1)

        with self.assertRaises(ErrorInitFrame):
            f1 = Frame(dict(a=3, b=4))

        with self.assertRaises(ErrorInitFrame):
            f1 = Frame(None, index=range(3), columns=range(3))

    def test_frame_init_w(self) -> None:
        f1 = Frame.from_dict(dict(a=(1, 2), b=(3, 4)), index=('x', 'y'))
        f2 = Frame(f1)
        self.assertEqual(f1._blocks.mloc.tolist(), f2._blocks.mloc.tolist())
        self.assertEqualFrames(f1, f2)

        f3 = Frame(f1, index=IndexAutoFactory, columns=IndexAutoFactory)
        self.assertEqual(f3.to_pairs(), ((0, ((0, 1), (1, 2))), (1, ((0, 3), (1, 4)))))
        self.assertEqual(f1._blocks.mloc.tolist(), f3._blocks.mloc.tolist())

        f4 = FrameGO(f1, index=('p', 'q'))
        f4['c'] = None
        self.assertEqual(
            f4.to_pairs(),
            (
                ('a', (('p', 1), ('q', 2))),
                ('b', (('p', 3), ('q', 4))),
                ('c', (('p', None), ('q', None))),
            ),
        )
        # first two values are stil equal
        self.assertEqual(f1._blocks.mloc.tolist(), f4._blocks.mloc.tolist()[:2])

        f5 = Frame(f4)
        self.assertEqual(
            f5.to_pairs(),
            (
                ('a', (('p', 1), ('q', 2))),
                ('b', (('p', 3), ('q', 4))),
                ('c', (('p', None), ('q', None))),
            ),
        )
        self.assertTrue(f5.columns.STATIC)

    def test_frame_init_x(self) -> None:
        f = Frame(columns=(), index=(3, 5))
        self.assertEqual(f.shape, (2, 0))
        self.assertEqual(f.to_pairs(), ())

    def test_frame_init_y(self) -> None:
        f1 = Frame(index=IndexAutoFactory(3))
        self.assertEqual(f1.shape, (3, 0))
        self.assertEqual(f1.to_pairs(), ())

        f2 = FrameGO(index=IndexAutoFactory(2))
        f2['a'] = (3, 9)
        f2['b'] = (4, 5)
        self.assertEqual(f2.index._map, None)  # type: ignore
        self.assertEqual(
            f2.to_pairs(), (('a', ((0, 3), (1, 9))), ('b', ((0, 4), (1, 5))))
        )

        f3 = Frame(
            np.arange(12).reshape(3, 4),
            index=IndexAutoFactory(3),
            columns=IndexAutoFactory(4),
        )
        self.assertEqual(
            f3.to_pairs(),
            (
                (0, ((0, 0), (1, 4), (2, 8))),
                (1, ((0, 1), (1, 5), (2, 9))),
                (2, ((0, 2), (1, 6), (2, 10))),
                (3, ((0, 3), (1, 7), (2, 11))),
            ),
        )

    # ---------------------------------------------------------------------------

    def test_frame_init_index_constructor_a(self) -> None:
        f1 = sf.Frame.from_element(
            'q',
            index=[('a', 'b'), (1, 2)],
            columns=tuple('xy'),
            index_constructor=IndexHierarchy.from_labels,
        )
        self.assertTrue(isinstance(f1.index, IndexHierarchy))
        self.assertEqual(
            f1.to_pairs(),
            (
                ('x', ((('a', 'b'), 'q'), ((1, 2), 'q'))),
                ('y', ((('a', 'b'), 'q'), ((1, 2), 'q'))),
            ),
        )

        with self.assertRaises(RuntimeError):
            f1 = sf.Frame.from_element(
                'q',
                index=[('a', 'b'), (1, 2)],
                columns=tuple('xy'),
                index_constructor=IndexHierarchyGO.from_labels,
            )

    def test_frame_init_columns_constructor_a(self) -> None:
        # using from_priduct is awkard, as it does not take a single iterable of products, but multiple args; we can get around this with a simple lambda
        f1 = sf.Frame.from_element(
            'q',
            index=tuple('xy'),
            columns=[('a', 'b'), (1, 2)],
            columns_constructor=lambda args: IndexHierarchy.from_product(*args),
        )
        self.assertTrue(isinstance(f1.columns, IndexHierarchy))
        self.assertEqual(
            f1.to_pairs(),
            (
                (('a', 1), (('x', 'q'), ('y', 'q'))),
                (('a', 2), (('x', 'q'), ('y', 'q'))),
                (('b', 1), (('x', 'q'), ('y', 'q'))),
                (('b', 2), (('x', 'q'), ('y', 'q'))),
            ),
        )

        with self.assertRaises(RuntimeError):
            f1 = sf.Frame.from_element(
                'q',
                index=tuple('xy'),
                columns=[('a', 'b'), (1, 2)],
                columns_constructor=lambda args: IndexHierarchyGO.from_product(*args),
            )

    def test_frame_init_iter(self) -> None:
        f1 = Frame.from_element(None, index=iter(range(3)), columns=('A',))
        self.assertEqual(f1.to_pairs(), (('A', ((0, None), (1, None), (2, None))),))

        f2 = Frame.from_element(None, index=('A',), columns=iter(range(3)))
        self.assertEqual(
            f2.to_pairs(),
            ((0, (('A', None),)), (1, (('A', None),)), (2, (('A', None),))),
        )

    def test_frame_values_a(self) -> None:
        f = sf.Frame.from_records([[3]])
        self.assertEqual(f.values.tolist(), [[3]])

    def test_frame_values_b(self) -> None:
        f = sf.Frame(np.array([[3, 2, 1]]))
        self.assertEqual(f.values.tolist(), [[3, 2, 1]])

    def test_frame_values_c(self) -> None:
        f = sf.Frame(np.array([[3], [2], [1]]))
        self.assertEqual(f.values.tolist(), [[3], [2], [1]])

    def test_frame_values_d(self) -> None:
        f = Frame.from_fields(
            (['2020-02', '1843-04'], [True, False], ['a', 'b']),
            dtypes=(np.datetime64, bool, str),
        )
        post = f.values
        self.assertEqual(
            post.tolist(),
            [
                [np.datetime64('2020-02'), True, 'a'],
                [np.datetime64('1843-04'), False, 'b'],
            ],
        )

    # ---------------------------------------------------------------------------

    def test_frame_from_series_a(self) -> None:
        s1 = Series((False, True, False), index=tuple('abc'))
        f1 = Frame.from_series(s1, name='foo')

        self.assertEqual(
            f1.to_pairs(), ((None, (('a', False), ('b', True), ('c', False))),)
        )

    def test_frame_from_series_b(self) -> None:
        s1 = Series((False, True, False), index=tuple('abc'), name='2018-05')
        f1 = Frame.from_series(s1, name='foo', columns_constructor=IndexYearMonth)
        self.assertEqual(f1.columns.__class__, IndexYearMonth)
        self.assertEqual(
            f1.to_pairs(),
            ((np.datetime64('2018-05'), (('a', False), ('b', True), ('c', False))),),
        )

    def test_frame_from_series_c(self) -> None:
        f1 = Frame.from_element(None, index=tuple('abc'), columns=('a',))
        with self.assertRaises(RuntimeError):
            f2 = Frame.from_series(f1)  # type: ignore

    # ---------------------------------------------------------------------------

    def test_frame_from_element_a(self) -> None:
        f1 = Frame.from_element(0, index=('a', 'b'), columns=('x', 'y', 'z'))
        self.assertEqual(f1.shape, (2, 3))
        self.assertEqual(
            f1.to_pairs(),
            (
                ('x', (('a', 0), ('b', 0))),
                ('y', (('a', 0), ('b', 0))),
                ('z', (('a', 0), ('b', 0))),
            ),
        )

    def test_frame_from_element_b(self) -> None:
        f1 = Frame.from_element(
            '2019', index=('a', 'b'), columns=('x', 'y', 'z'), dtype='datetime64[Y]'
        )
        self.assertEqual(f1.shape, (2, 3))
        self.assertEqual(
            f1.to_pairs(),
            (
                ('x', (('a', np.datetime64('2019')), ('b', np.datetime64('2019')))),
                ('y', (('a', np.datetime64('2019')), ('b', np.datetime64('2019')))),
                ('z', (('a', np.datetime64('2019')), ('b', np.datetime64('2019')))),
            ),
        )

    def test_frame_from_element_c(self) -> None:
        # not an error to create 0-sized frames
        f1 = Frame.from_element(
            '2019',
            index=('a', 'b'),
            columns=(),
        )
        self.assertEqual(f1.shape, (2, 0))

        f2 = Frame.from_element(
            '2019',
            index=(),
            columns=('x', 'y', 'z'),
        )
        self.assertEqual(f2.shape, (0, 3))

    def test_frame_from_element_d(self) -> None:
        idx1 = Index(('a', 'b'))
        idx2 = Index((3, 4))
        f1 = Frame.from_element(
            'x', index=idx1, columns=idx2, own_index=True, own_columns=True
        )
        self.assertTrue(id(idx1) == id(f1.index))
        self.assertTrue(id(idx2) == id(f1.columns))

    def test_frame_from_element_e(self) -> None:
        f1 = Frame.from_element(range(3), index=('a', 'b'), columns=('x', 'y', 'z'))
        self.assertEqual(f1.shape, (2, 3))
        self.assertEqual(
            f1.to_pairs(),
            (
                ('x', (('a', range(3)), ('b', range(3)))),
                ('y', (('a', range(3)), ('b', range(3)))),
                ('z', (('a', range(3)), ('b', range(3)))),
            ),
        )

        f2 = Frame.from_element([0], index=('a', 'b'), columns=('x', 'y', 'z'))
        self.assertEqual(f2.shape, (2, 3))
        self.assertEqual(
            f2.to_pairs(),
            (
                ('x', (('a', [0]), ('b', [0]))),
                ('y', (('a', [0]), ('b', [0]))),
                ('z', (('a', [0]), ('b', [0]))),
            ),
        )

    def test_frame_from_element_f(self) -> None:
        # not an error to create 0-sized frames
        with self.assertRaises(ErrorInitIndex):
            f1 = sf.Frame.from_element(1, index=range(5), columns='bar')

    # ---------------------------------------------------------------------------

    def test_frame_from_elements_a(self) -> None:
        f1 = Frame.from_elements(['a', 3, 'b'])
        self.assertEqual(f1.to_pairs(), ((0, ((0, 'a'), (1, 3), (2, 'b'))),))

        f2 = Frame.from_elements(['a', 3, 'b'], index=tuple('xyz'))
        self.assertEqual(f2.to_pairs(), ((0, (('x', 'a'), ('y', 3), ('z', 'b'))),))

        f3 = Frame.from_elements(['a', 3, 'b'], index=tuple('xyz'), columns=('p',))
        self.assertEqual(f3.to_pairs(), (('p', (('x', 'a'), ('y', 3), ('z', 'b'))),))

    def test_frame_from_elements_b(self) -> None:
        f1 = Frame.from_elements([5, False, 'X'])
        self.assertEqual(f1.to_pairs(), ((0, ((0, 5), (1, False), (2, 'X'))),))

        f2 = Frame.from_elements([5, False, 'X'], columns=('a', 'b'))
        self.assertEqual(
            f2.to_pairs(),
            (
                ('a', ((0, 5), (1, False), (2, 'X'))),
                ('b', ((0, 5), (1, False), (2, 'X'))),
            ),
        )

    def test_frame_from_elements_c(self) -> None:
        idx1 = Index(('a', 'b'))
        idx2 = Index((3, 4))
        f1 = Frame.from_elements(
            (10, 20), index=idx1, columns=idx2, own_index=True, own_columns=True
        )

        self.assertEqual(
            f1.to_pairs(), ((3, (('a', 10), ('b', 20))), (4, (('a', 10), ('b', 20))))
        )
        self.assertTrue(id(idx1) == id(f1.index))
        self.assertTrue(id(idx2) == id(f1.columns))

    # ---------------------------------------------------------------------------

    def test_frame_from_pairs_a(self) -> None:
        frame = Frame.from_items(sorted(dict(a=[3, 4, 5], b=[6, 3, 2]).items()))
        self.assertEqual(
            list((k, list(v.items())) for k, v in frame.items()),
            [('a', [(0, 3), (1, 4), (2, 5)]), ('b', [(0, 6), (1, 3), (2, 2)])],
        )

        frame = Frame.from_items(
            OrderedDict((('b', [6, 3, 2]), ('a', [3, 4, 5]))).items()
        )
        self.assertEqual(
            list((k, list(v.items())) for k, v in frame.items()),
            [('b', [(0, 6), (1, 3), (2, 2)]), ('a', [(0, 3), (1, 4), (2, 5)])],
        )

    # ---------------------------------------------------------------------------

    def test_frame_from_pandas_a(self) -> None:
        import pandas as pd

        df = pd.DataFrame(dict(a=(1, 2), b=(3, 4)))
        df.name = 'foo'

        f = Frame.from_pandas(df)
        self.assertEqual(f.to_pairs(), (('a', ((0, 1), (1, 2))), ('b', ((0, 3), (1, 4)))))

    def test_frame_from_pandas_b(self) -> None:
        import pandas as pd

        df = pd.DataFrame(dict(a=(1, 2), b=(False, True)), index=('x', 'y'))

        f = Frame.from_pandas(df)

        self.assertEqual(
            f.to_pairs(),
            (('a', (('x', 1), ('y', 2))), ('b', (('x', False), ('y', True)))),
        )

        with self.assertRaises(Exception):
            f['c'] = 0  # type: ignore

    def test_frame_from_pandas_c(self) -> None:
        import pandas as pd

        df = pd.DataFrame(dict(a=(1, 2), b=(False, True)), index=('x', 'y'))

        f = FrameGO.from_pandas(df)
        f['c'] = -1

        self.assertEqual(
            f.to_pairs(),
            (
                ('a', (('x', 1), ('y', 2))),
                ('b', (('x', False), ('y', True))),
                ('c', (('x', -1), ('y', -1))),
            ),
        )

    def test_frame_from_pandas_d(self) -> None:
        import pandas as pd

        df = pd.DataFrame(dict(a=(1, 2), b=(True, False)))
        df.name = 'foo'

        f = Frame.from_pandas(df, own_data=True)

        self.assertEqual(
            f.to_pairs(), (('a', ((0, 1), (1, 2))), ('b', ((0, True), (1, False))))
        )

    def test_frame_from_pandas_e(self) -> None:
        import pandas as pd

        df = pd.DataFrame(dict(a=(1, 2), b=(3, 4)), index=('x', 'y'))
        f = Frame.from_pandas(df, own_data=True, consolidate_blocks=True)
        self.assertEqual(
            f.to_pairs(), (('a', (('x', 1), ('y', 2))), ('b', (('x', 3), ('y', 4))))
        )

        self.assertEqual(f._blocks.shapes.tolist(), [(2, 2)])

    def test_frame_from_pandas_f(self) -> None:
        import pandas as pd

        df = pd.DataFrame(dict(a=(1, 2), b=('3', '4'), c=(1.5, 2.5), d=('a', 'b')))

        if hasattr(df, 'convert_dtypes'):
            df = df.convert_dtypes()

        df.name = 'foo'

        f = Frame.from_pandas(df)
        self.assertEqual(
            f.to_pairs(),
            (
                ('a', ((0, 1), (1, 2))),
                ('b', ((0, '3'), (1, '4'))),
                ('c', ((0, 1.5), (1, 2.5))),
                ('d', ((0, 'a'), (1, 'b'))),
            ),
        )

    @skip_win
    def test_frame_from_pandas_g(self) -> None:
        import pandas as pd

        df = pd.DataFrame(dict(a=(1, 2), b=(1.5, 2.5), c=('3', '4'), d=('a', 'b')))

        if hasattr(df, 'convert_dtypes'):
            df = df.convert_dtypes()
            f = Frame.from_pandas(df)
            self.assertEqual(
                f.dtypes.to_pairs(),
                (
                    ('a', np.dtype('int64')),
                    ('b', np.dtype('float64')),
                    ('c', np.dtype('<U1')),
                    ('d', np.dtype('<U1')),
                ),
            )
        else:
            f = Frame.from_pandas(df)
            self.assertEqual(
                f.dtypes.to_pairs(),
                (
                    ('a', np.dtype('int64')),
                    ('b', np.dtype('float64')),
                    ('c', np.dtype('O')),
                    ('d', np.dtype('O')),
                ),
            )

    def test_frame_from_pandas_h(self) -> None:
        import pandas as pd

        df = pd.DataFrame(
            dict(
                a=(False, True),
                b=(True, False),
                c=('q', 'r'),
                d=('a', 'b'),
                e=(False, False),
            )
        )

        if hasattr(df, 'convert_dtypes'):
            df = df.convert_dtypes()
            f = Frame.from_pandas(df)
            self.assertEqual(
                f.dtypes.to_pairs(),
                (
                    ('a', np.dtype('bool')),
                    ('b', np.dtype('bool')),
                    ('c', np.dtype('<U1')),
                    ('d', np.dtype('<U1')),
                    ('e', np.dtype('bool')),
                ),
            )
        else:
            f = Frame.from_pandas(df)
            # we do not have a chance to re-evaluate string types, so they come in as object
            self.assertEqual(
                f.dtypes.to_pairs(),
                (
                    ('a', np.dtype('bool')),
                    ('b', np.dtype('bool')),
                    ('c', np.dtype('O')),
                    ('d', np.dtype('O')),
                    ('e', np.dtype('bool')),
                ),
            )

    def test_frame_from_pandas_i(self) -> None:
        import pandas as pd

        df = pd.DataFrame(
            dict(
                a=(False, True),
                b=(True, np.nan),
                c=('q', 'r'),
                d=('a', np.nan),
                e=(False, False),
            )
        )

        if hasattr(df, 'convert_dtypes'):
            df = df.convert_dtypes()
            f = Frame.from_pandas(df)
            self.assertEqual(
                f.dtypes.to_pairs(),
                (
                    ('a', np.dtype('O')),
                    ('b', np.dtype('O')),
                    ('c', np.dtype('O')),
                    ('d', np.dtype('O')),
                    ('e', np.dtype('bool')),
                ),
            )
        else:
            f = Frame.from_pandas(df)
            self.assertEqual(
                f.dtypes.to_pairs(),
                (
                    ('a', np.dtype('bool')),
                    ('b', np.dtype('O')),
                    ('c', np.dtype('O')),
                    ('d', np.dtype('O')),
                    ('e', np.dtype('bool')),
                ),
            )

    def test_frame_from_pandas_j(self) -> None:
        import pandas as pd

        df = pd.DataFrame(dict(a=(1, 2), b=('3', '4'), c=(1.5, 2.5), d=('a', 'b')))

        f = Frame.from_pandas(df, index=IndexAutoFactory, columns=IndexAutoFactory)

        self.assertTrue(f.index._map is None)  # type: ignore
        self.assertTrue(f.columns._map is None)  # type: ignore

        self.assertEqual(
            f.to_pairs(),
            (
                (0, ((0, 1), (1, 2))),
                (1, ((0, '3'), (1, '4'))),
                (2, ((0, 1.5), (1, 2.5))),
                (3, ((0, 'a'), (1, 'b'))),
            ),
        )

    def test_frame_from_pandas_k(self) -> None:
        import pandas as pd

        df = pd.DataFrame.from_records(
            [(1, 2), ('3', '4'), (1.5, 2.5), ('a', 'b')],
            index=('2012', '2013', '2014', '2015'),
            columns=('2020-01', '2020-02'),
        )

        f = Frame.from_pandas(
            df, index_constructor=IndexYear, columns_constructor=IndexYearMonth
        )

        self.assertEqual(
            f.to_pairs(),
            (
                (
                    np.datetime64('2020-01'),
                    (
                        (np.datetime64('2012'), 1),
                        (np.datetime64('2013'), '3'),
                        (np.datetime64('2014'), 1.5),
                        (np.datetime64('2015'), 'a'),
                    ),
                ),
                (
                    np.datetime64('2020-02'),
                    (
                        (np.datetime64('2012'), 2),
                        (np.datetime64('2013'), '4'),
                        (np.datetime64('2014'), 2.5),
                        (np.datetime64('2015'), 'b'),
                    ),
                ),
            ),
        )

    def test_frame_from_pandas_m(self) -> None:
        import pandas as pd

        df = pd.DataFrame.from_records(
            [(1, 2), (3, 4), (1.5, 2.5)],
            index=pd.date_range('2012-01-01', '2012-01-03'),
        )
        f = sf.Frame.from_pandas(df, index_constructor=IndexDate)
        self.assertTrue(f.index.__class__ is IndexDate)

        self.assertEqual(
            f.to_pairs(),
            (
                (
                    0,
                    (
                        (np.datetime64('2012-01-01'), 1.0),
                        (np.datetime64('2012-01-02'), 3.0),
                        (np.datetime64('2012-01-03'), 1.5),
                    ),
                ),
                (
                    1,
                    (
                        (np.datetime64('2012-01-01'), 2.0),
                        (np.datetime64('2012-01-02'), 4.0),
                        (np.datetime64('2012-01-03'), 2.5),
                    ),
                ),
            ),
        )

    def test_frame_from_pandas_n(self) -> None:
        import pandas as pd

        df = pd.DataFrame(
            np.arange(8).reshape(2, 4),
            columns=pd.MultiIndex.from_product((('a', 'b'), (1, 2))),
        )

        ih1 = IndexHierarchy.from_pandas(df.columns)
        self.assertEqual(ih1.values.tolist(), [['a', 1], ['a', 2], ['b', 1], ['b', 2]])

        f = sf.Frame.from_pandas(df)
        self.assertEqual(
            f.to_pairs(),
            (
                (('a', 1), ((0, 0), (1, 4))),
                (('a', 2), ((0, 1), (1, 5))),
                (('b', 1), ((0, 2), (1, 6))),
                (('b', 2), ((0, 3), (1, 7))),
            ),
        )

        # this index shows itself as having only one level
        pd_idx = pd.MultiIndex([[]], [[]])
        with self.assertRaises(ErrorInitIndex):
            _ = sf.IndexHierarchy.from_pandas(pd_idx)

    def test_frame_from_pandas_o(self) -> None:
        f1 = Frame.from_records(
            [(1, 2), ('3', '4'), (1.5, 2.5), ('a', 'b')],
            index=('2012', '2013', '2014', '2015'),
            columns=('2020-01', '2020-02'),
        )
        with self.assertRaises(ErrorInitFrame):
            Frame.from_pandas(f1)

    def test_frame_from_pandas_p(self) -> None:
        import pandas as pd

        df = pd.DataFrame(dict(a=(1, 2), b=(True, False)))
        df.name = 'foo'

        f1 = Frame.from_pandas(df, name='bar')
        self.assertEqual(f1.name, 'bar')

        f2 = Frame.from_pandas(df)
        self.assertEqual(f2.name, 'foo')

        # can override name attr on DF
        f3 = Frame.from_pandas(df, name=None)
        self.assertEqual(f3.name, None)

    def test_frame_from_pandas_q(self) -> None:
        import pandas as pd

        df = pd.DataFrame(
            [
                [1960, '001002', 9900000.0],
                [1961, '001000', 900000.0],
                [1962, '001002', 800000.0],
            ],
            columns=['col1', 'col2', 0],
        )
        f = Frame.from_pandas(df)
        self.assertEqual(f.shape, df.shape)
        self.assertEqual(f.values.tolist(), df.values.tolist())

    def test_frame_from_pandas_r(self) -> None:
        import pandas as pd

        df = pd.DataFrame(
            [
                [1960, '001002', 9900000.0, 1.0],
                [1961, '001000', 900000.0, 2.0],
                [1962, '001002', 800000.0, 3.0],
            ],
            columns=['col1', 'col2', 0, 10],
        )

        f = Frame.from_pandas(df)
        self.assertEqual(f.shape, df.shape)
        self.assertEqual(f.values.tolist(), df.values.tolist())

    def test_frame_from_pandas_s(self) -> None:
        import pandas as pd

        df = pd.DataFrame(dict(a=(1, 2), b=('3', '4'), c=(True, False), d=('a', 'b')))
        f1 = Frame.from_pandas(df, dtypes={'c': int})
        self.assertEqual([d.kind for d in f1.dtypes.values], ['i', 'O', 'i', 'O'])

        f2 = Frame.from_pandas(df, dtypes=[bool, np.dtype('U1'), int, np.dtype('U1')])
        self.assertEqual([d.kind for d in f2.dtypes.values], ['b', 'U', 'i', 'U'])

    def test_frame_from_pandas_t(self) -> None:
        import pandas as pd

        df = pd.DataFrame(np.arange(20).reshape(2, 10) % 2)

        f1 = Frame.from_pandas(df, dtypes={3: bool})
        self.assertEqual(
            [d.kind for d in f1.dtypes.values],
            ['i', 'i', 'i', 'b', 'i', 'i', 'i', 'i', 'i', 'i'],
        )

        f2 = Frame.from_pandas(df, dtypes=bool)
        self.assertEqual(
            [d.kind for d in f2.dtypes.values],
            ['b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b'],
        )

        f3 = Frame.from_pandas(df, dtypes={3: bool, 5: float})
        self.assertEqual(
            [d.kind for d in f3.dtypes.values],
            ['i', 'i', 'i', 'b', 'i', 'f', 'i', 'i', 'i', 'i'],
        )

    def test_frame_from_pandas_u(self) -> None:
        import pandas as pd

        df = pd.DataFrame(dict(a=(1, 0)))
        f1 = Frame.from_pandas(df, dtypes=bool, own_data=True)
        self.assertEqual(f1.shape, (2, 1))
        self.assertEqual([d.kind for d in f1.dtypes.values], ['b'])

    def test_frame_from_pandas_v(self) -> None:
        import pandas as pd

        df = pd.DataFrame(columns=list('abc'))  # size 0
        f1 = Frame.from_pandas(df, dtypes=bool)
        self.assertEqual(
            f1.dtypes.values.tolist(),
            [np.dtype('bool'), np.dtype('bool'), np.dtype('bool')],
        )

    def test_frame_from_pandas_w(self) -> None:
        f1 = ff.parse('s(2,2)|c(IH,(str,str))|i(IH,(int,int))|v(bool)')
        df = f1.to_pandas()
        f2 = Frame.from_pandas(df)
        self.assertEqual(f2.columns.depth, 2)
        self.assertEqual(f2.index.depth, 2)
        self.assertEqual(
            f2.to_pairs(),
            (
                (
                    ('zZbu', 'zOyq'),
                    (((34715, 105269), False), ((34715, 119909), False)),
                ),
                (
                    ('zZbu', 'zIA5'),
                    (((34715, 105269), False), ((34715, 119909), False)),
                ),
            ),
        )

    # ---------------------------------------------------------------------------

    def test_frame_to_pandas_a(self) -> None:
        records = (
            (1, 2, 'a', False),
            (30, 34, 'b', True),
            (54, 95, 'c', False),
            (65, 73, 'd', True),
        )
        columns = IndexHierarchy.from_product(('a', 'b'), (1, 2))
        index = IndexHierarchy.from_product((100, 200), (True, False))
        f1 = Frame.from_records(records, columns=columns, index=index)

        df = f1.to_pandas()

        self.assertEqual(
            df.index.values.tolist(),
            [(100, True), (100, False), (200, True), (200, False)],
        )

        self.assertEqual(
            df.columns.values.tolist(), [('a', 1), ('a', 2), ('b', 1), ('b', 2)]
        )

        self.assertEqual(
            df.values.tolist(),
            [
                [1, 2, 'a', False],
                [30, 34, 'b', True],
                [54, 95, 'c', False],
                [65, 73, 'd', True],
            ],
        )

    def test_frame_to_pandas_b(self) -> None:
        f1 = sf.Frame.from_dict_records(
            [dict(a=1, b=1), dict(a=2, b=3), dict(a=1, b=1), dict(a=2, b=3)],
            index=sf.IndexHierarchy.from_labels(
                [(1, 'dd', 0), (1, 'b', 0), (2, 'cc', 0), (2, 'ee', 0)]
            ),
        )

        df = f1.loc[sf.HLoc[(1, ['dd'])]].to_pandas()

        self.assertEqual(df.columns.values.tolist(), ['a', 'b'])
        self.assertEqual(df.values.tolist(), [[1, 1]])
        self.assertEqual(df.index.values.tolist(), [(1, 'dd', 0)])

    def test_frame_to_pandas_c(self) -> None:
        f = sf.FrameGO.from_elements(['a' for x in range(5)], columns=['a'])
        f['b'] = [1.0 for i in range(5)]
        df = f.to_pandas()
        self.assertEqual(df.dtypes.tolist(), [np.dtype(object), np.dtype(np.float64)])

    @skip_win
    def test_frame_to_pandas_d(self) -> None:
        records = (
            (1, 2, 'a', False),
            (30, 34, 'b', True),
            (54, 95, 'c', False),
            (65, 73, 'd', True),
        )
        columns = IndexHierarchy.from_product(('a', 'b'), (1, 2))
        index = IndexHierarchy.from_product((100, 200), (True, False))
        f1 = Frame.from_records(records, columns=columns, index=index)

        df = f1.to_pandas()

        self.assertEqual(
            df.dtypes.tolist(),
            [np.dtype('int64'), np.dtype('int64'), np.dtype('O'), np.dtype('bool')],
        )

    def test_frame_to_pandas_e(self) -> None:
        f = Frame.from_records(
            [['a', 1, 10], ['a', 2, 200], ['b', 1, -3], ['b', 2, 7]],
            columns=('x', 'y', 'z'),
        )
        df = f.set_index_hierarchy(['x', 'y']).to_pandas()
        self.assertEqual(list(df.index.names), ['x', 'y'])

    def test_frame_to_pandas_f(self) -> None:
        # check name transfer
        f = Frame.from_records(
            [
                ['a', 1, 10],
                ['a', 2, 200],
            ],
            columns=('x', 'y', 'z'),
            name='foo',
        )
        df = f.to_pandas()
        self.assertEqual(df.name, f.name)

    def test_frame_to_pandas_g(self) -> None:
        # check single block
        f = Frame(np.arange(2000).reshape(100, 20))
        self.assertTrue(f._blocks.unified)

        df = f.to_pandas()
        self.assertEqual(df.shape, (100, 20))

    def test_frame_to_pandas_h(self) -> None:
        df = Frame().to_pandas()
        self.assertEqual(df.shape, (0, 0))

    def test_frame_to_pandas_i(self) -> None:
        df = FrameGO(index=range(10)).to_pandas()
        self.assertEqual(df.shape, (10, 0))

    def test_frame_to_pandas_j(self) -> None:
        df = Frame.from_element('', columns=range(10), index=()).to_pandas()
        self.assertEqual(df.shape, (0, 10))

    # ---------------------------------------------------------------------------

    def test_frame_to_arrow_a(self) -> None:
        records = (
            (1, 2, 'a', False),
            (30, 34, 'b', True),
            (54, 95, 'c', False),
            (65, 73, 'd', True),
        )
        columns = IndexHierarchy.from_product(('a', 'b'), (1, 2))
        index = IndexHierarchy.from_product((100, 200), (True, False))
        f1 = Frame.from_records(records, columns=columns, index=index)

        at = f1.to_arrow()
        self.assertEqual(at.shape, (4, 6))
        self.assertEqual(
            at.column_names,
            ['__index0__', '__index1__', "['a' 1]", "['a' 2]", "['b' 1]", "['b' 2]"],
        )
        self.assertEqual(
            at.to_pydict(),
            {
                '__index0__': [100, 100, 200, 200],
                '__index1__': [True, False, True, False],
                "['a' 1]": [1, 30, 54, 65],
                "['a' 2]": [2, 34, 95, 73],
                "['b' 1]": ['a', 'b', 'c', 'd'],
                "['b' 2]": [False, True, False, True],
            },
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
        f1 = Frame.from_records(records, columns=columns, index=index)
        at = f1.to_arrow()

        f2 = Frame.from_arrow(
            at, index_depth=f1.index.depth, columns_depth=f1.columns.depth
        )
        # String arrays will come in as objects
        self.assertEqualFrames(f1, f2, compare_dtype=False)

    def test_frame_from_arrow_b(self) -> None:
        records = (
            (1, 2, 'a', False),
            (30, 34, 'b', True),
            (54, 95, 'c', False),
            (65, 73, 'd', True),
        )
        columns = IndexHierarchy.from_product(('a', 'b'), (1, 2))
        index = IndexHierarchy.from_product((100, 200), (True, False))
        f1 = Frame.from_records(records, columns=columns, index=index)
        at = f1.to_arrow()

        f2 = Frame.from_arrow(
            at,
            index_depth=f1.index.depth,
            columns_depth=f1.columns.depth,
            consolidate_blocks=True,
        )
        self.assertEqual(f2._blocks.shapes.tolist(), [(4, 2), (4,), (4,)])

    def test_frame_from_arrow_c(self) -> None:
        records = (
            (1, 2, 'a', False),
            (30, 34, 'b', True),
            (54, 95, 'c', False),
            (65, 73, 'd', True),
        )
        f1 = Frame.from_records(records)
        at = f1.to_arrow(include_index=False, include_columns=False)
        f2 = Frame.from_arrow(at, index_depth=0, columns_depth=0, consolidate_blocks=True)

        self.assertEqual(
            f2.to_pairs(),
            (
                (0, ((0, 1), (1, 30), (2, 54), (3, 65))),
                (1, ((0, 2), (1, 34), (2, 95), (3, 73))),
                (2, ((0, 'a'), (1, 'b'), (2, 'c'), (3, 'd'))),
                (3, ((0, False), (1, True), (2, False), (3, True))),
            ),
        )

    def test_frame_from_arrow_d(self) -> None:
        records = (
            (1, 2, 'a', False),
            (30, 34, 'b', True),
            (54, 95, 'c', False),
            (65, 73, 'd', True),
        )
        f1 = Frame.from_records(records)
        f1 = f1.set_index(0, drop=True)
        at = f1.to_arrow(include_index=True, include_columns=False)
        f2 = Frame.from_arrow(at, index_depth=1, columns_depth=0, consolidate_blocks=True)
        self.assertEqual(
            f2.to_pairs(),
            (
                (0, ((1, 2), (30, 34), (54, 95), (65, 73))),
                (1, ((1, 'a'), (30, 'b'), (54, 'c'), (65, 'd'))),
                (2, ((1, False), (30, True), (54, False), (65, True))),
            ),
        )

    # ---------------------------------------------------------------------------

    def test_frame_to_parquet_a(self) -> None:
        records = (
            (1, 2, 'a', False),
            (30, 34, 'b', True),
            (54, 95, 'c', False),
            (65, 73, 'd', True),
        )
        columns = IndexHierarchy.from_product(('a', 'b'), (1, 2))
        index = IndexHierarchy.from_product((100, 200), (True, False))
        f1 = Frame.from_records(records, columns=columns, index=index)

        with temp_file('.parquet') as fp:
            f1.to_parquet(fp)

    def test_frame_to_parquet_b(self) -> None:
        records = (
            (1, 2, 'a', False),
            (30, 34, 'b', True),
            (54, 95, 'c', False),
            (65, 73, 'd', True),
        )
        columns = IndexHierarchy.from_product(('a', 'b'), (1, 2))
        index = IndexHierarchy.from_product((100, 200), (True, False))
        f1 = Frame.from_records(records, columns=columns, index=index)

        with temp_file('.parquet') as fp:
            f1.to_parquet(fp, include_index=False, include_columns=False)
            f2 = Frame.from_parquet(fp)

        self.assertEqual(
            f2.to_pairs(),
            (
                ('0', ((0, 1), (1, 30), (2, 54), (3, 65))),
                ('1', ((0, 2), (1, 34), (2, 95), (3, 73))),
                ('2', ((0, 'a'), (1, 'b'), (2, 'c'), (3, 'd'))),
                ('3', ((0, False), (1, True), (2, False), (3, True))),
            ),
        )

    def test_frame_to_parquet_c(self) -> None:
        records = (
            (1, 'a', False),
            (30, 'b', True),
            (54, 'c', False),
            (65, 'd', True),
        )
        index = IndexDate.from_date_range('2017-12-15', '2017-12-18')
        f1 = FrameGO.from_records(records, columns=('a', 'b', 'c'))
        f1['d'] = index.values

        with temp_file('.parquet') as fp:
            f1.to_parquet(fp, include_index=False, include_columns=True)
            f2 = Frame.from_parquet(fp, columns_depth=1)

        self.assertEqual(
            f2.to_pairs(),
            (
                ('a', ((0, 1), (1, 30), (2, 54), (3, 65))),
                ('b', ((0, 'a'), (1, 'b'), (2, 'c'), (3, 'd'))),
                ('c', ((0, False), (1, True), (2, False), (3, True))),
                (
                    'd',
                    (
                        (0, np.datetime64('2017-12-15T00:00:00.000000000')),
                        (1, np.datetime64('2017-12-16T00:00:00.000000000')),
                        (2, np.datetime64('2017-12-17T00:00:00.000000000')),
                        (3, np.datetime64('2017-12-18T00:00:00.000000000')),
                    ),
                ),
            ),
        )
        self.assertTrue(f2.index._map is None)  # type: ignore

    def test_frame_to_parquet_d(self) -> None:
        # pyarrow.lib.ArrowNotImplementedError: Unsupported datetime64 time unit

        f1 = Frame(
            IndexDate.from_date_range('2017-12-15', '2017-12-18')
            .values.astype('datetime64[ns]')
            .reshape(2, 2)
        )
        with temp_file('.parquet') as fp:
            f1.to_parquet(fp, include_index=False, include_columns=True)
            f2 = Frame.from_parquet(fp, columns_depth=1)

        self.assertEqual(
            f2.to_pairs(),
            (
                (
                    '0',
                    (
                        (0, np.datetime64('2017-12-15T00:00:00.000000000')),
                        (1, np.datetime64('2017-12-17T00:00:00.000000000')),
                    ),
                ),
                (
                    '1',
                    (
                        (0, np.datetime64('2017-12-16T00:00:00.000000000')),
                        (1, np.datetime64('2017-12-18T00:00:00.000000000')),
                    ),
                ),
            ),
        )

        f3 = Frame(
            IndexDate.from_date_range('2017-12-15', '2017-12-18')
            .values.astype('datetime64[D]')
            .reshape(2, 2)
        )
        with temp_file('.parquet') as fp:
            f3.to_parquet(fp, include_index=False, include_columns=True)
            f4 = Frame.from_parquet(fp, columns_depth=1)

        self.assertEqual(
            f4.to_pairs(),
            (
                (
                    '0',
                    (
                        (0, np.datetime64('2017-12-15T00:00:00.000000000')),
                        (1, np.datetime64('2017-12-17T00:00:00.000000000')),
                    ),
                ),
                (
                    '1',
                    (
                        (0, np.datetime64('2017-12-16T00:00:00.000000000')),
                        (1, np.datetime64('2017-12-18T00:00:00.000000000')),
                    ),
                ),
            ),
        )

        f5 = Frame(
            IndexDate.from_date_range('2017-12-15', '2017-12-18')
            .values.astype('datetime64[s]')
            .reshape(2, 2)
        )
        with temp_file('.parquet') as fp:
            f5.to_parquet(fp, include_index=False, include_columns=True)
            f6 = Frame.from_parquet(fp, columns_depth=1)

        self.assertEqual(
            f6.to_pairs(),
            (
                (
                    '0',
                    (
                        (0, np.datetime64('2017-12-15T00:00:00.000000000')),
                        (1, np.datetime64('2017-12-17T00:00:00.000000000')),
                    ),
                ),
                (
                    '1',
                    (
                        (0, np.datetime64('2017-12-16T00:00:00.000000000')),
                        (1, np.datetime64('2017-12-18T00:00:00.000000000')),
                    ),
                ),
            ),
        )

    def test_frame_to_parquet_e(self) -> None:
        # pyarrow.lib.ArrowNotImplementedError: Unsupported datetime64 time unit

        f7 = Frame(
            IndexDate.from_date_range('2017-12-15', '2017-12-18')
            .values.astype('datetime64[m]')
            .reshape(2, 2)
        )
        with temp_file('.parquet') as fp:
            f7.to_parquet(fp, include_index=False, include_columns=True)
            f8 = Frame.from_parquet(fp, columns_depth=1)

        self.assertEqual(
            f8.to_pairs(),
            (
                (
                    '0',
                    (
                        (0, np.datetime64('2017-12-15T00:00:00.000000000')),
                        (1, np.datetime64('2017-12-17T00:00:00.000000000')),
                    ),
                ),
                (
                    '1',
                    (
                        (0, np.datetime64('2017-12-16T00:00:00.000000000')),
                        (1, np.datetime64('2017-12-18T00:00:00.000000000')),
                    ),
                ),
            ),
        )

        f5 = Frame(
            IndexDate.from_date_range('2017-12-15', '2017-12-18')
            .values.astype('datetime64[h]')
            .reshape(2, 2)
        )

        with temp_file('.parquet') as fp:
            f5.to_parquet(fp, include_index=False, include_columns=True)
            f6 = Frame.from_parquet(fp, columns_depth=1)

        self.assertEqual(
            f6.to_pairs(),
            (
                (
                    '0',
                    (
                        (0, np.datetime64('2017-12-15T00:00:00.000000000')),
                        (1, np.datetime64('2017-12-17T00:00:00.000000000')),
                    ),
                ),
                (
                    '1',
                    (
                        (0, np.datetime64('2017-12-16T00:00:00.000000000')),
                        (1, np.datetime64('2017-12-18T00:00:00.000000000')),
                    ),
                ),
            ),
        )

        f3 = Frame(
            IndexDate.from_date_range('2017-12-15', '2017-12-18')
            .values.astype('datetime64[M]')
            .reshape(2, 2)
        )

        with temp_file('.parquet') as fp:
            f3.to_parquet(fp, include_index=False, include_columns=True)
            f4 = Frame.from_parquet(fp, columns_depth=1)

        self.assertEqual(
            f4.to_pairs(),
            (
                (
                    '0',
                    (
                        (0, np.datetime64('2017-12-01T00:00:00.000000000')),
                        (1, np.datetime64('2017-12-01T00:00:00.000000000')),
                    ),
                ),
                (
                    '1',
                    (
                        (0, np.datetime64('2017-12-01T00:00:00.000000000')),
                        (1, np.datetime64('2017-12-01T00:00:00.000000000')),
                    ),
                ),
            ),
        )

        f1 = Frame(
            IndexDate.from_date_range('2017-12-15', '2017-12-18')
            .values.astype('datetime64[Y]')
            .reshape(2, 2)
        )

        with temp_file('.parquet') as fp:
            f1.to_parquet(fp, include_index=False, include_columns=True)
            f2 = Frame.from_parquet(fp, columns_depth=1)

        self.assertEqual(
            f2.to_pairs(),
            (
                (
                    '0',
                    (
                        (0, np.datetime64('2017-01-01T00:00:00.000000000')),
                        (1, np.datetime64('2017-01-01T00:00:00.000000000')),
                    ),
                ),
                (
                    '1',
                    (
                        (0, np.datetime64('2017-01-01T00:00:00.000000000')),
                        (1, np.datetime64('2017-01-01T00:00:00.000000000')),
                    ),
                ),
            ),
        )

    def test_frame_to_parquet_f(self) -> None:
        records = (
            (1, 2, 'a', False),
            (30, 34, 'b', True),
            (54, 95, 'c', False),
            (65, 73, 'd', True),
        )
        columns = Index(tuple('ABCD'), name='foo')
        index = Index(tuple('WXYZ'), name='bar')
        f1 = Frame.from_records(records, columns=columns, index=index)

        with temp_file('.parquet') as fp:
            with self.assertRaises(RuntimeError):
                f1.to_parquet(
                    fp,
                    include_index=True,
                    include_columns=True,
                    include_index_name=True,
                    include_columns_name=True,
                )

            f1.to_parquet(
                fp,
                include_index=True,
                include_columns=True,
                include_index_name=True,
            )
            f2 = Frame.from_parquet(
                fp,
                index_depth=1,
                index_name_depth_level=0,
                columns_depth=1,
            )
            self.assertEqual(f2.index.name, 'bar')
            self.assertEqual(f2.columns.name, None)

            f3 = Frame.from_parquet(
                fp,
                index_depth=1,
                columns_depth=1,
                columns_name_depth_level=0,
            )
            self.assertEqual(f3.index.name, None)
            self.assertEqual(f3.columns.name, 'bar')

            f4 = Frame.from_parquet(
                fp,
                index_depth=1,
                columns_depth=1,
                index_name_depth_level=0,
                columns_name_depth_level=0,
            )
            self.assertEqual(f4.index.name, 'bar')
            self.assertEqual(f4.columns.name, 'bar')

    # ---------------------------------------------------------------------------

    def test_frame_from_parquet_a(self) -> None:
        records = (
            (1, 2, 'a', False),
            (30, 34, 'b', True),
            (54, 95, 'c', False),
            (65, 73, 'd', True),
        )
        columns = IndexHierarchy.from_product(('a', 'b'), (1, 2))
        index = IndexHierarchy.from_product((100, 200), (True, False))
        f1 = Frame.from_records(records, columns=columns, index=index)

        with temp_file('.parquet') as fp:
            f1.to_parquet(fp)
            f2 = Frame.from_parquet(
                fp, index_depth=f1.index.depth, columns_depth=f1.columns.depth
            )

        self.assertEqualFrames(f1, f2, compare_dtype=False)

    def test_frame_from_parquet_b1(self) -> None:
        records = (
            (1, 2, 'a', False),
            (30, 34, 'b', True),
            (54, 95, 'c', False),
            (65, 73, 'd', True),
        )
        columns = ('a', 'b', 'c', 'd')
        f1 = Frame.from_records(
            records,
            columns=columns,
        )

        with temp_file('.parquet') as fp:
            f1.to_parquet(fp)

            with self.assertRaises(ErrorInitFrame):
                _ = Frame.from_parquet(
                    fp,
                    index_depth=1,  # cannot use with columns_selectg
                    columns_select=('d', 'a'),
                    columns_depth=1,
                )

            f2 = Frame.from_parquet(
                fp, index_depth=0, columns_select=('d', 'a'), columns_depth=1
            )

        self.assertEqual(
            f2.to_pairs(),
            (
                ('d', ((0, False), (1, True), (2, False), (3, True))),
                ('a', ((0, 1), (1, 30), (2, 54), (3, 65))),
            ),
        )

        self.assertTrue(f2.index._map is None)  # type: ignore

    def test_frame_from_parquet_b2(self) -> None:
        records = (
            (1, 2, 'a', False),
            (30, 34, 'b', True),
            (54, 95, 'c', False),
            (65, 73, 'd', True),
        )
        columns = ('a', 'b', 'c', 'd')
        f1 = Frame.from_records(
            records,
            columns=columns,
        )

        with temp_file('.parquet') as fp:
            f1.to_parquet(fp)

            # prove we raise if columns_select as columns not found
            # might be pyarrow.lib.ArrowInvalid or ErrorInitFrame
            with self.assertRaises(Exception):
                f2 = Frame.from_parquet(
                    fp, index_depth=0, columns_select=('d', 'foo'), columns_depth=1
                )

    def test_frame_from_parquet_c(self) -> None:
        f = sf.FrameGO.from_element(
            'a', index=range(3), columns=sf.IndexHierarchy.from_labels((('a', 'b'),))
        )

        with temp_file('.parquet') as fp:
            f.to_parquet(fp)
            f1 = sf.Frame.from_parquet(fp, index_depth=1, columns_depth=2)
            # strings come back as object
            self.assertTrue(f.equals(f1, compare_dtype=False, compare_class=False))

            # when index_depth is not provided an exception is raised
            with self.assertRaises(RuntimeError):
                sf.Frame.from_parquet(fp, columns_depth=2)

    @skip_win
    def test_frame_from_parquet_d(self) -> None:
        dt64 = np.datetime64
        dtype = np.dtype

        items = (
            ('a', (dt64('2020-01-01'), dt64('2019-02-02'))),
            ('b', ('foo', 'bar')),
            ('c', (True, False)),
            ('d', (2000, 3000)),
        )
        index = ('x', 'y')
        f1 = Frame.from_items(
            items,
            index=index,
        )

        with temp_file('.parquet') as fp:
            f1.to_parquet(fp)
            f2 = Frame.from_parquet(fp, index_depth=1, columns_depth=1)

            self.assertEqual([dt.kind for dt in f2.dtypes.values], ['M', 'O', 'b', 'i'])

            f3 = Frame.from_parquet(
                fp,
                index_depth=1,
                columns_depth=1,
                dtypes={'a': 'datetime64[D]', 'b': str},
            )
            self.assertEqual(
                f3.dtypes.values.tolist(),
                [dtype('<M8[D]'), dtype('<U3'), dtype('bool'), dtype('int64')],
            )

            # positional dtypes include the index array
            f4 = Frame.from_parquet(
                fp,
                index_depth=1,
                columns_depth=1,
                dtypes=(None, 'datetime64[Y]', str, bool, '<U4'),
            )

            self.assertEqual(
                f4.dtypes.values.tolist(),
                [dtype('<M8[Y]'), dtype('<U3'), dtype('bool'), dtype('<U4')],
            )

            # if dtypes is an iterable, it has to be of length equal to data records
            with self.assertRaises(IndexError):
                f5 = Frame.from_parquet(
                    fp,
                    index_depth=1,
                    columns_depth=1,
                    dtypes=(None, 'datetime64[Y]', str, bool),
                )

            # dtypes can take a single type
            f5 = Frame.from_parquet(fp, index_depth=1, columns_depth=1, dtypes=str)
            self.assertEqual([dt.kind for dt in f5.dtypes.values], ['U', 'U', 'U', 'U'])

    def test_frame_from_parquet_e(self) -> None:
        dt64 = np.datetime64
        dtype = np.dtype

        records = (
            (10.1, 20.1, False, dt64('2020-01-01')),
            (-5.1, 0.1, True, dt64('2000-09-01')),
            (2000.1, 33.1, False, dt64('2017-03-01')),
        )
        f1 = Frame.from_records(records)

        with temp_file('.parquet') as fp:
            f1.to_parquet(fp, include_index=False, include_columns=False)
            f2 = Frame.from_parquet(fp, index_depth=0, columns_depth=0)

            self.assertEqual([dt.kind for dt in f2.dtypes.values], ['f', 'f', 'b', 'M'])

            # can include fields that are not used; this does not raise
            f3 = Frame.from_parquet(
                fp, index_depth=0, columns_depth=0, dtypes={'foo': str}
            )

            f4 = Frame.from_parquet(
                fp,
                index_depth=0,
                columns_depth=0,
                dtypes=(float, float, str, 'datetime64[Y]'),
            )

            self.assertEqual(
                f4.dtypes.values.tolist(),
                [dtype('float64'), dtype('float64'), dtype('<U5'), dtype('<M8[Y]')],
            )

            self.assertEqual(
                f4.to_pairs(),
                (
                    (0, ((0, 10.1), (1, -5.1), (2, 2000.1))),
                    (1, ((0, 20.1), (1, 0.1), (2, 33.1))),
                    (2, ((0, 'False'), (1, 'True'), (2, 'False'))),
                    (3, ((0, dt64('2020')), (1, dt64('2000')), (2, dt64('2017')))),
                ),
            )

    def test_frame_from_parquet_f(self) -> None:
        # arrow was segfaulting on None; we identify and raise
        with self.assertRaises(ValueError):
            f1 = Frame.from_parquet(None)

    # ---------------------------------------------------------------------------

    def test_frame_to_xarray_a(self) -> None:
        records = (
            (1, 2, 'a', False),
            (30, 34, 'b', True),
            (54, 95, 'c', False),
            (65, 73, 'd', True),
        )
        columns = IndexHierarchy.from_product(('a', 'b'), (1, 2), name=('a', 'b'))
        index = IndexHierarchy.from_labels(
            ((200, False, 'a'), (200, True, 'b'), (100, False, 'a'), (300, True, 'b'))
        )

        f1 = Frame.from_records(records, columns=columns, index=index)
        ds1 = f1.to_xarray()
        self.assertEqual(
            tuple(ds1.data_vars.keys()), (('a', 1), ('a', 2), ('b', 1), ('b', 2))
        )
        self.assertEqual(
            tuple(ds1.coords.keys()), ('__index0__', '__index1__', '__index2__')
        )
        self.assertEqual(ds1[('b', 1)].values.ndim, 3)

        f2 = Frame.from_records(records)
        ds2 = f2.to_xarray()
        self.assertEqual(tuple(ds2.data_vars.keys()), (0, 1, 2, 3))
        self.assertEqual(tuple(ds2.coords.keys()), ('__index0__',))
        self.assertEqual(ds2[3].values.tolist(), [False, True, False, True])

    # ---------------------------------------------------------------------------

    def test_frame_to_series_a(self) -> None:
        f1 = ff.parse('s(4,5)|i(I,str)|c(I,int)')

        s1 = f1.to_series()
        self.assertEqual(
            s1.to_pairs(),
            (
                (('zZbu', 34715), 1930.4),
                (('zZbu', -3648), -610.8),
                (('zZbu', 91301), 694.3),
                (('zZbu', 30205), 1080.4),
                (('zZbu', 54020), 3511.58),
                (('ztsv', 34715), -1760.34),
                (('ztsv', -3648), 3243.94),
                (('ztsv', 91301), -72.96),
                (('ztsv', 30205), 2580.34),
                (('ztsv', 54020), 1175.36),
                (('zUvW', 34715), 1857.34),
                (('zUvW', -3648), -823.14),
                (('zUvW', 91301), 1826.02),
                (('zUvW', 30205), 700.42),
                (('zUvW', 54020), 2925.68),
                (('zkuW', 34715), 1699.34),
                (('zkuW', -3648), 114.58),
                (('zkuW', 91301), 604.1),
                (('zkuW', 30205), 3338.48),
                (('zkuW', 54020), 3408.8),
            ),
        )

        s2 = f1.to_series(index_constructor=IndexHierarchy.from_labels)
        self.assertEqual(
            s2.to_pairs(),
            (
                (('zZbu', 34715), 1930.4),
                (('zZbu', -3648), -610.8),
                (('zZbu', 91301), 694.3),
                (('zZbu', 30205), 1080.4),
                (('zZbu', 54020), 3511.58),
                (('ztsv', 34715), -1760.34),
                (('ztsv', -3648), 3243.94),
                (('ztsv', 91301), -72.96),
                (('ztsv', 30205), 2580.34),
                (('ztsv', 54020), 1175.36),
                (('zUvW', 34715), 1857.34),
                (('zUvW', -3648), -823.14),
                (('zUvW', 91301), 1826.02),
                (('zUvW', 30205), 700.42),
                (('zUvW', 54020), 2925.68),
                (('zkuW', 34715), 1699.34),
                (('zkuW', -3648), 114.58),
                (('zkuW', 91301), 604.1),
                (('zkuW', 30205), 3338.48),
                (('zkuW', 54020), 3408.8),
            ),
        )

    def test_frame_to_series_b(self) -> None:
        f1 = ff.parse('s(4,5)|i(IH,(str,str))|c(I,int)')
        s1 = f1.to_series()
        self.assertEqual(
            s1.to_pairs(),
            (
                (('zZbu', 'zOyq', 34715), 1930.4),
                (('zZbu', 'zOyq', -3648), -610.8),
                (('zZbu', 'zOyq', 91301), 694.3),
                (('zZbu', 'zOyq', 30205), 1080.4),
                (('zZbu', 'zOyq', 54020), 3511.58),
                (('zZbu', 'zIA5', 34715), -1760.34),
                (('zZbu', 'zIA5', -3648), 3243.94),
                (('zZbu', 'zIA5', 91301), -72.96),
                (('zZbu', 'zIA5', 30205), 2580.34),
                (('zZbu', 'zIA5', 54020), 1175.36),
                (('ztsv', 'zGDJ', 34715), 1857.34),
                (('ztsv', 'zGDJ', -3648), -823.14),
                (('ztsv', 'zGDJ', 91301), 1826.02),
                (('ztsv', 'zGDJ', 30205), 700.42),
                (('ztsv', 'zGDJ', 54020), 2925.68),
                (('ztsv', 'zmhG', 34715), 1699.34),
                (('ztsv', 'zmhG', -3648), 114.58),
                (('ztsv', 'zmhG', 91301), 604.1),
                (('ztsv', 'zmhG', 30205), 3338.48),
                (('ztsv', 'zmhG', 54020), 3408.8),
            ),
        )

        s2 = f1.to_series(index_constructor=IndexHierarchy.from_labels)
        self.assertEqual(
            s2.to_pairs(),
            (
                (('zZbu', 'zOyq', 34715), 1930.4),
                (('zZbu', 'zOyq', -3648), -610.8),
                (('zZbu', 'zOyq', 91301), 694.3),
                (('zZbu', 'zOyq', 30205), 1080.4),
                (('zZbu', 'zOyq', 54020), 3511.58),
                (('zZbu', 'zIA5', 34715), -1760.34),
                (('zZbu', 'zIA5', -3648), 3243.94),
                (('zZbu', 'zIA5', 91301), -72.96),
                (('zZbu', 'zIA5', 30205), 2580.34),
                (('zZbu', 'zIA5', 54020), 1175.36),
                (('ztsv', 'zGDJ', 34715), 1857.34),
                (('ztsv', 'zGDJ', -3648), -823.14),
                (('ztsv', 'zGDJ', 91301), 1826.02),
                (('ztsv', 'zGDJ', 30205), 700.42),
                (('ztsv', 'zGDJ', 54020), 2925.68),
                (('ztsv', 'zmhG', 34715), 1699.34),
                (('ztsv', 'zmhG', -3648), 114.58),
                (('ztsv', 'zmhG', 91301), 604.1),
                (('ztsv', 'zmhG', 30205), 3338.48),
                (('ztsv', 'zmhG', 54020), 3408.8),
            ),
        )

    def test_frame_to_series_c(self) -> None:
        f1 = ff.parse('s(4,5)|i(IH,(str,str))|c(IH,(int,int))').rename('foo')
        s1 = f1.to_series()
        self.assertEqual(s1.name, f1.name)
        self.assertEqual(
            s1.to_pairs(),
            (
                (('zZbu', 'zOyq', 34715, 105269), 1930.4),
                (('zZbu', 'zOyq', 34715, 119909), -610.8),
                (('zZbu', 'zOyq', -3648, 194224), 694.3),
                (('zZbu', 'zOyq', -3648, 172133), 1080.4),
                (('zZbu', 'zOyq', 91301, 96520), 3511.58),
                (('zZbu', 'zIA5', 34715, 105269), -1760.34),
                (('zZbu', 'zIA5', 34715, 119909), 3243.94),
                (('zZbu', 'zIA5', -3648, 194224), -72.96),
                (('zZbu', 'zIA5', -3648, 172133), 2580.34),
                (('zZbu', 'zIA5', 91301, 96520), 1175.36),
                (('ztsv', 'zGDJ', 34715, 105269), 1857.34),
                (('ztsv', 'zGDJ', 34715, 119909), -823.14),
                (('ztsv', 'zGDJ', -3648, 194224), 1826.02),
                (('ztsv', 'zGDJ', -3648, 172133), 700.42),
                (('ztsv', 'zGDJ', 91301, 96520), 2925.68),
                (('ztsv', 'zmhG', 34715, 105269), 1699.34),
                (('ztsv', 'zmhG', 34715, 119909), 114.58),
                (('ztsv', 'zmhG', -3648, 194224), 604.1),
                (('ztsv', 'zmhG', -3648, 172133), 3338.48),
                (('ztsv', 'zmhG', 91301, 96520), 3408.8),
            ),
        )

        s2 = f1.to_series(index_constructor=IndexHierarchy.from_labels)
        self.assertEqual(s2.index.depth, 4)

        self.assertEqual(
            s2.to_pairs(),
            (
                (('zZbu', 'zOyq', 34715, 105269), 1930.4),
                (('zZbu', 'zOyq', 34715, 119909), -610.8),
                (('zZbu', 'zOyq', -3648, 194224), 694.3),
                (('zZbu', 'zOyq', -3648, 172133), 1080.4),
                (('zZbu', 'zOyq', 91301, 96520), 3511.58),
                (('zZbu', 'zIA5', 34715, 105269), -1760.34),
                (('zZbu', 'zIA5', 34715, 119909), 3243.94),
                (('zZbu', 'zIA5', -3648, 194224), -72.96),
                (('zZbu', 'zIA5', -3648, 172133), 2580.34),
                (('zZbu', 'zIA5', 91301, 96520), 1175.36),
                (('ztsv', 'zGDJ', 34715, 105269), 1857.34),
                (('ztsv', 'zGDJ', 34715, 119909), -823.14),
                (('ztsv', 'zGDJ', -3648, 194224), 1826.02),
                (('ztsv', 'zGDJ', -3648, 172133), 700.42),
                (('ztsv', 'zGDJ', 91301, 96520), 2925.68),
                (('ztsv', 'zmhG', 34715, 105269), 1699.34),
                (('ztsv', 'zmhG', 34715, 119909), 114.58),
                (('ztsv', 'zmhG', -3648, 194224), 604.1),
                (('ztsv', 'zmhG', -3648, 172133), 3338.48),
                (('ztsv', 'zmhG', 91301, 96520), 3408.8),
            ),
        )

    # ---------------------------------------------------------------------------

    def test_frame_getitem_a(self) -> None:
        records = ((1, 2, 'a', False, True), (30, 50, 'b', True, False))

        f1 = Frame.from_records(
            records, columns=('p', 'q', 'r', 's', 't'), index=('x', 'y')
        )

        f2: Frame = f1['r':]  # type: ignore
        self.assertEqual(f2.columns.values.tolist(), ['r', 's', 't'])
        self.assertTrue((f2.index == f1.index).all())
        self.assertEqual(mloc(f2.index.values), mloc(f1.index.values))

    def test_frame_getitem_b(self) -> None:
        records = ((1, 2, 'a', False, True), (30, 50, 'b', True, False))

        f1 = Frame.from_records(
            records, columns=('p', 'q', 'r', 's', 't'), index=('x', 'y')
        )

        # using an Index object for selection
        self.assertEqual(
            f1[f1.columns.loc['r':]].to_pairs(),  # type: ignore  # https://github.com/python/typeshed/pull/3024
            (
                ('r', (('x', 'a'), ('y', 'b'))),
                ('s', (('x', False), ('y', True))),
                ('t', (('x', True), ('y', False))),
            ),
        )

    def test_frame_getitem_c(self) -> None:
        records = ((1, 2, 'a', False), (30, 50, 'b', True))

        f1 = FrameGO.from_records(
            records,
            columns=IndexHierarchyGO.from_product(('A', 'B'), (1, 2)),
            index=('x', 'y'),
        )

        # we can use a tuple to select a single column if a hierarchical index
        self.assertEqual(f1[('A', 2)].to_pairs(), (('x', 2), ('y', 50)))

    def test_frame_getitem_d(self) -> None:
        records = ((1, 2, 'a', False), (30, 50, 'b', True))

        f1 = FrameGO.from_records(
            records,
            columns=Index([('A', 1), ('A', 2), ('B', 1), ('B', 2)]),
            index=('x', 'y'),
        )

        self.assertEqual(f1[('A', 2)].to_pairs(), (('x', 2), ('y', 50)))
        self.assertEqual(
            f1[[('A', 2), ('B', 1)]].to_pairs(),
            ((('A', 2), (('x', 2), ('y', 50))), (('B', 1), (('x', 'a'), ('y', 'b')))),
        )

    def test_frame_getitem_e(self) -> None:
        f = sf.Frame()
        with self.assertRaises(KeyError):
            _ = f['a']

    # ---------------------------------------------------------------------------

    def test_frame_length_a(self) -> None:
        records = ((1, 2, 'a', False, True), (30, 50, 'b', True, False))

        f1 = Frame.from_records(
            records, columns=('p', 'q', 'r', 's', 't'), index=('x', 'y')
        )

        self.assertEqual(len(f1), 2)

    # ---------------------------------------------------------------------------

    def test_frame_iloc_a(self) -> None:
        records = ((1, 2, 'a', False, True), (30, 50, 'b', True, False))

        f1 = Frame.from_records(
            records, columns=('p', 'q', 'r', 's', 't'), index=('x', 'y')
        )

        self.assertEqual((f1.iloc[0].values == f1.loc['x'].values).all(), True)
        self.assertEqual((f1.iloc[1].values == f1.loc['y'].values).all(), True)

    def test_frame_iloc_b(self) -> None:
        # this is example derived from this question:
        # https://stackoverflow.com/questions/22927181/selecting-specific-rows-and-columns-from-numpy-array

        a = np.arange(20).reshape((5, 4))
        f1 = FrameGO(a)
        a[1, 1] = 3000  # ensure we made a copy
        self.assertEqual(
            f1.loc[[0, 1, 3], [0, 2]].values.tolist(), [[0, 2], [4, 6], [12, 14]]
        )
        self.assertEqual(
            f1.iloc[[0, 1, 3], [0, 2]].values.tolist(), [[0, 2], [4, 6], [12, 14]]
        )

        self.assertTrue(f1._index._map is None)  # type: ignore
        self.assertTrue(f1._columns._map is None)

        f1[4] = list(range(5))
        self.assertTrue(f1._columns._map is None)

        f1[20] = list(range(5))
        self.assertFalse(f1._columns._map is None)

        self.assertEqual(
            f1.values.tolist(),
            [
                [0, 1, 2, 3, 0, 0],
                [4, 5, 6, 7, 1, 1],
                [8, 9, 10, 11, 2, 2],
                [12, 13, 14, 15, 3, 3],
                [16, 17, 18, 19, 4, 4],
            ],
        )

    # ---------------------------------------------------------------------------

    def test_frame_setitem_a(self) -> None:
        records = ((1, 2, 'a', False, True), (30, 50, 'b', True, False))

        f1 = FrameGO.from_records(
            records, columns=('p', 'q', 'r', 's', 't'), index=('x', 'y')
        )

        f1['a'] = (False, True)
        self.assertEqual(f1['a'].values.tolist(), [False, True])

        # test index alginment
        f1['b'] = Series((3, 2, 5), index=('y', 'x', 'g'))
        self.assertEqual(f1['b'].values.tolist(), [2, 3])

        f1['c'] = Series((300, 200, 500), index=('y', 'j', 'k'))
        self.assertAlmostEqualItems(f1['c'].items(), [('x', nan), ('y', 300)])

    def test_frame_setitem_b(self) -> None:
        records = ((1, 2, 'a', False, True), (30, 50, 'b', True, False))

        f1 = FrameGO.from_records(
            records, columns=('p', 'q', 'r', 's', 't'), index=('x', 'y')
        )

        f1['u'] = 0

        self.assertEqual(f1.loc['x'].values.tolist(), [1, 2, 'a', False, True, 0])

        # with self.assertRaises(Exception):
        f1['w'] = [[1, 2], [4, 5]]
        self.assertEqual(f1['w'].to_pairs(), (('x', [1, 2]), ('y', [4, 5])))

    def test_frame_setitem_c(self) -> None:
        f1 = FrameGO(index=sf.Index(tuple('abcde')))
        f1['a'] = 30
        self.assertEqual(
            f1.to_pairs(),
            (('a', (('a', 30), ('b', 30), ('c', 30), ('d', 30), ('e', 30))),),
        )

    def test_frame_setitem_d(self) -> None:
        # 3d array raises exception
        f = sf.FrameGO(index=range(3))
        f['a'] = 5
        self.assertEqual(f.sum().values[0], 15)

    def test_frame_setitem_e(self) -> None:
        # 3d array raises exception
        f = sf.FrameGO(index=range(3))
        f['a'] = 'foo'
        self.assertEqual(f.to_pairs(), (('a', ((0, 'foo'), (1, 'foo'), (2, 'foo'))),))

    def test_frame_setitem_f(self) -> None:
        # 3d array raises exception
        f = sf.FrameGO(index=range(3))
        f['a'] = 'foo'

        with self.assertRaises(RuntimeError):
            f['a'] = 'bar4'

    def test_frame_setitem_g(self) -> None:
        # 3d array raises exception
        f = sf.FrameGO(index=range(3))
        f['a'] = 'foo'

        # with self.assertRaises(RuntimeError):
        with self.assertRaises(RuntimeError):
            f['b'] = np.array([[1, 2], [2, 5]])

        with self.assertRaises(RuntimeError):
            f['b'] = np.array([1, 2])

        with self.assertRaises(RuntimeError):
            f['b'] = [1, 2]

    def test_frame_setitem_h(self) -> None:
        # 3d array raises exception
        f = sf.FrameGO.from_element(
            'a', index=range(3), columns=sf.IndexHierarchy.from_labels((('a', 1),))
        )

        f[sf.HLoc['a', 2]] = 3
        # this was resulting in a mal-formed blocks
        with self.assertRaises(RuntimeError):
            f[sf.HLoc['a', 2]] = False

        self.assertEqual(f.shape, (3, 2))
        self.assertEqual(f.columns.shape, (2, 2))
        self.assertEqual(
            f.to_pairs(),
            (
                (('a', 1), ((0, 'a'), (1, 'a'), (2, 'a'))),
                (('a', 2), ((0, 3), (1, 3), (2, 3))),
            ),
        )

    def test_frame_setitem_i(self) -> None:
        records = ((1, 2, 'a', False), (30, 50, 'b', True))

        f1 = FrameGO.from_records(records, columns=('p', 'q', 'r', 's'), index=('x', 'y'))

        with self.assertRaises(RuntimeError):
            f1['t'] = [1, 2, 4]

    def test_frame_setitem_j(self) -> None:
        records = ((1, 2, 'a', False), (30, 50, 'b', True))

        f1 = FrameGO.from_records(
            records,
            columns=IndexHierarchyGO.from_product(('A', 'B'), (1, 2)),
            index=('x', 'y'),
        )

        # set and retrieve with the same kye
        key = ('C', 1)
        f1[key] = 3
        post = f1[key]
        self.assertEqual(post.to_pairs(), (('x', 3), ('y', 3)))

        with self.assertRaises(RuntimeError):
            f1[('C', 2, 3)] = False

        with self.assertRaises(RuntimeError):
            f1[HLoc['C', 2, 3]] = False

        with self.assertRaises(RuntimeError):
            f1[('C',)] = False

        with self.assertRaises(RuntimeError):
            f1[HLoc['C',]] = False

        # can assign to a right-sized HLoc
        f1[HLoc['C', 2]] = False

        self.assertEqual(
            f1.to_pairs(),
            (
                (('A', 1), (('x', 1), ('y', 30))),
                (('A', 2), (('x', 2), ('y', 50))),
                (('B', 1), (('x', 'a'), ('y', 'b'))),
                (('B', 2), (('x', False), ('y', True))),
                (('C', 1), (('x', 3), ('y', 3))),
                (('C', 2), (('x', False), ('y', False))),
            ),
        )

    def test_frame_setitem_k(self) -> None:
        f1 = sf.FrameGO.from_records(np.arange(9).reshape(3, 3))

        def gen1() -> tp.Iterator[int]:
            yield 1
            raise ValueError('gen1')

        try:
            f1['a'] = gen1()
        except ValueError:
            pass

        self.assertEqual(f1.shape, (3, 3))
        self.assertEqual(len(f1.columns), 3)

    def test_frame_setitem_l1(self) -> None:
        f = sf.FrameGO(index=range(3))
        f['a'] = f.index
        self.assertEqual(f.to_pairs(), (('a', ((0, 0), (1, 1), (2, 2))),))

        with self.assertRaises(RuntimeError):
            f['b'] = sf.Series(['a', 'b']).index

    def test_frame_setitem_m(self) -> None:
        f1 = sf.FrameGO.from_records(np.arange(9).reshape(3, 3))

        def gen1(v: bool) -> tp.Iterator[int]:
            if v:
                raise ValueError('gen1')
            yield 1

        try:
            f1['a'] = gen1(True)
        except ValueError:
            pass

        self.assertEqual(f1.shape, (3, 3))
        self.assertEqual(len(f1.columns), 3)

    def test_frame_setitem_n(self) -> None:
        f = sf.FrameGO.from_element(
            'a', index=range(3), columns=sf.IndexHierarchy.from_labels((('a', 'b'),))
        )
        with self.assertRaises(RuntimeError):
            f[('s', 'b')] = f

    def test_frame_setitem_o(self) -> None:
        import pandas as pd

        # insure that you cannot set a Pandas Series inot a Frame

        pds = pd.Series(dict(a=2, b=3, c=4))

        f = sf.FrameGO(index=('c', 'b', 'a'))
        with self.assertRaises(RuntimeError):
            f['x'] = pds

        f['x'] = pds.values  # can apply array out of ordewr
        f['y'] = Series.from_pandas(pds)

        self.assertEqual(
            f.to_pairs(),
            (
                ('x', (('c', 2), ('b', 3), ('a', 4))),
                ('y', (('c', 4), ('b', 3), ('a', 2))),
            ),
        )

    def test_frame_setitem_p(self) -> None:
        f1 = FrameGO.from_dict(dict(a=[1, 2, 3], b=[4, 5, 6]))
        f1 = f1.relabel(columns=sf.IndexDate('2024-01-01 2024-01-02'.split()))
        f1['c'] = [7, 8, 9]
        self.assertEqual(
            f1.columns.values.tolist(),
            [np.datetime64('2024-01-01'), np.datetime64('2024-01-02'), 'c'],
        )

        self.assertEqual(f1.shape, (3, 3))

    # ---------------------------------------------------------------------------

    def test_frame_extend_items_a(self) -> None:
        records = ((1, 2, 'a', False, True), (30, 50, 'b', True, False))

        f1 = FrameGO.from_records(
            records, columns=('p', 'q', 'r', 's', 't'), index=('x', 'y')
        )

        columns = OrderedDict((('c', np.array([0, -1])), ('d', np.array([3, 5]))))

        f1.extend_items(columns.items())

        self.assertEqual(f1.columns.values.tolist(), ['p', 'q', 'r', 's', 't', 'c', 'd'])

        self.assertTypeBlocksArrayEqual(
            f1._blocks,
            [[1, 2, 'a', False, True, 0, 3], [30, 50, 'b', True, False, -1, 5]],
            match_dtype=object,
        )

    def test_frame_extend_a(self) -> None:
        records = ((1, 2, 'a', False, True), (30, 50, 'b', True, False))
        f1 = FrameGO.from_records(
            records, columns=('p', 'q', 'r', 's', 't'), index=('x', 'y')
        )

        blocks = (np.array([[50, 40], [30, 20]]), np.array([[50, 40], [30, 20]]))
        columns = ('a', 'b', 'c', 'd')
        f2 = Frame(TypeBlocks.from_blocks(blocks), columns=columns, index=('y', 'z'))

        f1.extend(f2)
        f3 = f1.fillna(None)

        self.assertEqual(
            f1.columns.values.tolist(), ['p', 'q', 'r', 's', 't', 'a', 'b', 'c', 'd']
        )

        self.assertEqual(
            f3.to_pairs(),
            (
                ('p', (('x', 1), ('y', 30))),
                ('q', (('x', 2), ('y', 50))),
                ('r', (('x', 'a'), ('y', 'b'))),
                ('s', (('x', False), ('y', True))),
                ('t', (('x', True), ('y', False))),
                ('a', (('x', None), ('y', 50))),
                ('b', (('x', None), ('y', 40))),
                ('c', (('x', None), ('y', 50))),
                ('d', (('x', None), ('y', 40))),
            ),
        )

    def test_frame_extend_b(self) -> None:
        records = (('a', False, True), ('b', True, False))
        f1 = FrameGO.from_records(records, columns=('p', 'q', 'r'), index=('x', 'y'))

        s1 = Series((200, -3), index=('y', 'x'))

        # this will work with a None name

        f1.extend(s1)

        self.assertEqual(f1.columns.values.tolist(), ['p', 'q', 'r', None])
        self.assertEqual(f1[None].values.tolist(), [-3, 200])

    def test_frame_extend_c(self) -> None:
        records = (('a', False, True), ('b', True, False))
        f1 = FrameGO.from_records(records, columns=('p', 'q', 'r'), index=('x', 'y'))

        s1 = Series((200, -3), index=('y', 'x'), name='s')

        f1.extend(s1)

        self.assertEqual(f1.columns.values.tolist(), ['p', 'q', 'r', 's'])
        self.assertEqual(f1['s'].values.tolist(), [-3, 200])

    def test_frame_extend_d(self) -> None:
        records = (('a', False, True), ('b', True, False))
        f1 = FrameGO.from_records(records, columns=('p', 'q', 'r'), index=('x', 'y'))

        s1 = Series((200, -3), index=('q', 'x'), name='s')

        f1.extend(s1, fill_value=0)

        self.assertEqual(f1.columns.values.tolist(), ['p', 'q', 'r', 's'])
        self.assertEqual(f1['s'].values.tolist(), [-3, 0])

    def test_frame_extend_e(self) -> None:
        records = (('a', False, True), ('b', True, False))
        f1 = FrameGO.from_records(records, columns=('p', 'q', 'r'), index=('x', 'y'))

        with self.assertRaises(NotImplementedError):
            f1.extend('a')

    def test_frame_extend_f(self) -> None:
        records = (('a', 'c', False, True), ('b', 'd', True, False))
        f1 = FrameGO.from_records(
            records,
            columns=IndexHierarchyGO.from_product(('A', 'B'), (1, 2)),
            index=('x', 'y'),
        )

        records = (('x', 'w', False, True), ('y', 'q', True, False))
        f2 = FrameGO.from_records(
            records,
            columns=IndexHierarchyGO.from_product(('C', 'D'), (1, 2)),
            index=('x', 'y'),
        )

        f1.extend(f2)
        self.assertEqual(
            f1.to_pairs(),
            (
                (('A', 1), (('x', 'a'), ('y', 'b'))),
                (('A', 2), (('x', 'c'), ('y', 'd'))),
                (('B', 1), (('x', False), ('y', True))),
                (('B', 2), (('x', True), ('y', False))),
                (('C', 1), (('x', 'x'), ('y', 'y'))),
                (('C', 2), (('x', 'w'), ('y', 'q'))),
                (('D', 1), (('x', False), ('y', True))),
                (('D', 2), (('x', True), ('y', False))),
            ),
        )

    def test_frame_extend_g(self) -> None:
        records = (('a', 'c', False, True), ('b', 'd', True, False))
        f1 = FrameGO.from_records(
            records,
            columns=IndexHierarchyGO.from_product(('A', 'B'), (1, 2)),
            index=('x', 'y'),
        )

        # extending with a non GO frame into a GO
        records = (('x', 'w', False, True), ('y', 'q', True, False))
        f2 = Frame.from_records(
            records,
            columns=IndexHierarchy.from_product(('C', 'D'), (1, 2)),
            index=('x', 'y'),
        )

        f1.extend(f2)
        self.assertEqual(
            f1.to_pairs(),
            (
                (('A', 1), (('x', 'a'), ('y', 'b'))),
                (('A', 2), (('x', 'c'), ('y', 'd'))),
                (('B', 1), (('x', False), ('y', True))),
                (('B', 2), (('x', True), ('y', False))),
                (('C', 1), (('x', 'x'), ('y', 'y'))),
                (('C', 2), (('x', 'w'), ('y', 'q'))),
                (('D', 1), (('x', False), ('y', True))),
                (('D', 2), (('x', True), ('y', False))),
            ),
        )

        self.assertEqual(f1.__class__, FrameGO)

    def test_frame_extend_h(self) -> None:
        records = (('a', 'c', False, True), ('b', 'd', True, False))
        f1 = FrameGO.from_records(
            records,
            columns=IndexHierarchyGO.from_product(('A', 'B'), (1, 2)),
            index=('x', 'y'),
        )

        s1 = sf.Series(('e', 'f'), index=('x', 'y'), name=('C', 1))
        f1.extend(s1)

        self.assertEqual(
            f1.to_pairs(),
            (
                (('A', 1), (('x', 'a'), ('y', 'b'))),
                (('A', 2), (('x', 'c'), ('y', 'd'))),
                (('B', 1), (('x', False), ('y', True))),
                (('B', 2), (('x', True), ('y', False))),
                (('C', 1), (('x', 'e'), ('y', 'f'))),
            ),
        )

    def test_frame_extend_i(self) -> None:
        f1 = FrameGO(index=('x', 'y'))
        records = (('a', False, True), ('b', True, False))
        f2 = Frame.from_records(records, columns=('p', 'q', 'r'), index=('x', 'y'))

        f1.extend(f2)

        self.assertEqual(
            f1.to_pairs(),
            (
                ('p', (('x', 'a'), ('y', 'b'))),
                ('q', (('x', False), ('y', True))),
                ('r', (('x', True), ('y', False))),
            ),
        )

    def test_frame_extend_j(self) -> None:
        records = (('a', False), ('b', True))
        f1 = FrameGO.from_records(records, columns=('p', 'q'), index=('x', 'y'))
        f2 = f1.relabel(columns=IndexDateGO(('2022-01-05', '2025-05-12')))
        f2.extend(f1)
        self.assertEqual(
            f2.columns.values.tolist(),
            [np.datetime64('2022-01-05'), np.datetime64('2025-05-12'), 'p', 'q'],
        )
        self.assertEqual(f2.shape, (2, 4))

    def test_frame_extend_k(self) -> None:
        records = (('a', False), ('b', True))
        f1 = FrameGO.from_records(
            records, columns=IndexDateGO(('2022-01-05', '2025-05-12')), index=('x', 'y')
        )
        s1 = Series((2, 10), index=(('x', 'y')), name='foo')
        f1.extend(s1)
        self.assertEqual(
            f1.columns.values.tolist(),
            [np.datetime64('2022-01-05'), np.datetime64('2025-05-12'), 'foo'],
        )
        self.assertEqual(f1.shape, (2, 3))

    # ---------------------------------------------------------------------------

    def test_frame_extend_empty_a(self) -> None:
        # full Frame, empty extensions with no index
        records = (
            (1, 2, 'a', False, True),
            (30, 34, 'b', True, False),
            (54, 95, 'c', False, False),
        )
        f1 = FrameGO.from_records(
            records, columns=('a', 'b', 'c', 'd', 'e'), index=('x', 'y', 'z')
        )

        f2 = FrameGO()  # no index or columns

        f1.extend(f2)
        self.assertEqual(f1.shape, (3, 5))  # extension happens, but no change in shape

    def test_frame_extend_empty_b(self) -> None:
        # full Frame, empty extension with index
        records = (
            (1, 2, 'a', False, True),
            (30, 34, 'b', True, False),
            (54, 95, 'c', False, False),
        )
        f1 = FrameGO.from_records(
            records, columns=('a', 'b', 'c', 'd', 'e'), index=('x', 'y', 'z')
        )

        f2 = FrameGO(index=('x', 'y', 'z'))
        f1.extend(f2)
        self.assertEqual(f1.shape, (3, 5))  # extension happens, but no change in shape

    def test_frame_extend_empty_c(self) -> None:
        # empty with index, full frame extension

        records = (
            (1, 2, 'a', False, True),
            (30, 34, 'b', True, False),
            (54, 95, 'c', False, False),
        )
        f1 = FrameGO.from_records(
            records, columns=('a', 'b', 'c', 'd', 'e'), index=('x', 'y', 'z')
        )

        f2 = FrameGO(index=('x', 'y', 'z'))
        f2.extend(f1)
        self.assertEqual(f2.shape, (3, 5))  # extension happens, but no change in shape

        self.assertEqual(
            f2.to_pairs(),
            (
                ('a', (('x', 1), ('y', 30), ('z', 54))),
                ('b', (('x', 2), ('y', 34), ('z', 95))),
                ('c', (('x', 'a'), ('y', 'b'), ('z', 'c'))),
                ('d', (('x', False), ('y', True), ('z', False))),
                ('e', (('x', True), ('y', False), ('z', False))),
            ),
        )

    def test_frame_extend_empty_d(self) -> None:
        # full Frame, empty extension with different index
        records = (
            (1, 2, 'a', False, True),
            (30, 34, 'b', True, False),
            (54, 95, 'c', False, False),
        )
        f1 = FrameGO.from_records(
            records, columns=('a', 'b', 'c', 'd', 'e'), index=('x', 'y', 'z')
        )

        f2 = FrameGO(index=('w', 'x', 'y', 'z'))

        f1.extend(f2)
        self.assertEqual(f1.shape, (3, 5))  # extension happens, but no change in shape

    def test_frame_extend_empty_e(self) -> None:
        # empty Frame with no index extended by full frame
        records = (
            (1, 2, 'a', False, True),
            (30, 34, 'b', True, False),
            (54, 95, 'c', False, False),
        )
        f1 = FrameGO.from_records(
            records, columns=('a', 'b', 'c', 'd', 'e'), index=('x', 'y', 'z')
        )

        f2 = FrameGO()  # no index or columns

        f2.extend(f1)
        # as we align on the caller's index, if that index is empty, there is nothing to take from the passed Frame; however, since we observe columns, we add those (empty columns). this falls out of lower-level implementations: could be done differently if desirable.
        self.assertEqual(f2.shape, (0, 5))

    def test_frame_extend_empty_f(self) -> None:
        f1 = FrameGO(columns=('p', 'q'))
        f2 = Frame(columns=('r', 's'))

        f1.extend(f2)
        self.assertEqual(f1.to_pairs(), (('p', ()), ('q', ()), ('r', ()), ('s', ())))

        s1 = sf.Series((), name='t')
        f1.extend(s1)
        self.assertEqual(
            f1.to_pairs(), (('p', ()), ('q', ()), ('r', ()), ('s', ()), ('t', ()))
        )

    # ---------------------------------------------------------------------------

    def test_frame_insert_a(self) -> None:
        records = ((1, 2, 'a', False, True), (30, 50, 'b', True, False))
        f1 = Frame.from_records(
            records, columns=('p', 'q', 'r', 's', 't'), index=('x', 'y')
        )

        blocks = (np.array([[50, 40], [30, 20]]), np.array([[50, 40], [30, 20]]))
        columns = ('a', 'b', 'c', 'd')
        f2 = Frame(TypeBlocks.from_blocks(blocks), columns=columns, index=('y', 'z'))

        f3 = f1._insert(2, f2, after=False, fill_value=None)

        self.assertEqual(
            f3.to_pairs(),
            (
                ('p', (('x', 1), ('y', 30))),
                ('q', (('x', 2), ('y', 50))),
                ('a', (('x', None), ('y', 50))),
                ('b', (('x', None), ('y', 40))),
                ('c', (('x', None), ('y', 50))),
                ('d', (('x', None), ('y', 40))),
                ('r', (('x', 'a'), ('y', 'b'))),
                ('s', (('x', False), ('y', True))),
                ('t', (('x', True), ('y', False))),
            ),
        )

    def test_frame_insert_b1(self) -> None:
        records = (('a', False, True), ('b', True, False))
        f1 = FrameGO.from_records(records, columns=('p', 'q', 'r'), index=('x', 'y'))

        with self.assertRaises(NotImplementedError):
            f1._insert(0, 'a', after=False)

        f_empty = sf.Frame(np.array(()).reshape(2, 0), index=f1.index)
        f2 = f1.insert_before('q', f_empty)
        self.assertTrue(f1.equals(f2))  # no insertion of no width frame
        self.assertNotEqual(id(f1), id(f2))

        f3 = Frame.from_records(records, columns=('p', 'q', 'r'), index=('x', 'y'))
        f4 = f3.insert_before('q', f_empty)
        self.assertTrue(f1.equals(f2))  # no insertion of no width frame
        self.assertEqual(id(f3), id(f4))

        # matching index but no columns
        f5 = FrameGO(columns=(), index=('x', 'y'))
        f6 = f3.insert_before('q', f5)
        self.assertTrue(f3.equals(f6))  # no insertion of no width frame
        self.assertEqual(id(f3), id(f6))

    def test_frame_insert_b2(self) -> None:
        records = (('a', False, True), ('b', True, False))
        f1 = FrameGO.from_records(records, columns=('p', 'q', 'r'), index=('x', 'y'))

        f_empty = sf.Frame(np.array(()).reshape(0, 2), columns=('s', 't'))
        f2 = f1.insert_before('q', f_empty)
        self.assertEqual(f2.columns.values.tolist(), ['p', 's', 't', 'q', 'r'])

    def test_frame_insert_c(self) -> None:
        records = (('a', False, True), ('b', True, False))
        f1 = FrameGO.from_records(records, columns=('p', 'q', 'r'), index=('x', 'y'))

        s1 = Series((200, -3), index=('y', 'x'), name='s')

        f2 = f1._insert(0, s1, after=False)

        self.assertEqual(
            f2.to_pairs(),
            (
                ('s', (('x', -3), ('y', 200))),
                ('p', (('x', 'a'), ('y', 'b'))),
                ('q', (('x', False), ('y', True))),
                ('r', (('x', True), ('y', False))),
            ),
        )

        f3 = f1._insert(iloc_to_insertion_iloc(-1, len(f1.columns)), s1, after=False)
        self.assertEqual(
            f3.to_pairs(),
            (
                ('p', (('x', 'a'), ('y', 'b'))),
                ('q', (('x', False), ('y', True))),
                ('s', (('x', -3), ('y', 200))),
                ('r', (('x', True), ('y', False))),
            ),
        )

        f4 = f1._insert(2, s1, after=True)  # same as appending
        self.assertEqual(
            f4.to_pairs(),
            (
                ('p', (('x', 'a'), ('y', 'b'))),
                ('q', (('x', False), ('y', True))),
                ('r', (('x', True), ('y', False))),
                ('s', (('x', -3), ('y', 200))),
            ),
        )

    def test_frame_insert_d(self) -> None:
        records = (
            (1, 2, 'a', False),
            (30, 34, 'b', True),
            (54, 95, 'c', False),
            (65, 73, 'd', True),
        )
        columns = IndexHierarchy.from_product(('a', 'b'), (1, 2))
        index = IndexHierarchy.from_product((100, 200), (True, False))
        f1 = Frame.from_records(records, columns=columns, index=index)

        f2 = Frame(np.arange(8).reshape(4, 2), index=index, columns=(('c', 1), ('c', 2)))

        f3 = f1._insert(2, f2, after=False)

        self.assertEqual(
            f3.to_pairs(),
            (
                (
                    ('a', 1),
                    (
                        ((100, True), 1),
                        ((100, False), 30),
                        ((200, True), 54),
                        ((200, False), 65),
                    ),
                ),
                (
                    ('a', 2),
                    (
                        ((100, True), 2),
                        ((100, False), 34),
                        ((200, True), 95),
                        ((200, False), 73),
                    ),
                ),
                (
                    ('c', 1),
                    (
                        ((100, True), 0),
                        ((100, False), 2),
                        ((200, True), 4),
                        ((200, False), 6),
                    ),
                ),
                (
                    ('c', 2),
                    (
                        ((100, True), 1),
                        ((100, False), 3),
                        ((200, True), 5),
                        ((200, False), 7),
                    ),
                ),
                (
                    ('b', 1),
                    (
                        ((100, True), 'a'),
                        ((100, False), 'b'),
                        ((200, True), 'c'),
                        ((200, False), 'd'),
                    ),
                ),
                (
                    ('b', 2),
                    (
                        ((100, True), False),
                        ((100, False), True),
                        ((200, True), False),
                        ((200, False), True),
                    ),
                ),
            ),
        )

    def test_frame_insert_e(self) -> None:
        records = (('a', False, True), ('b', True, False))
        f1 = FrameGO.from_records(records, columns=('p', 'q', 'r'), index=('x', 'y'))

        s1 = Series((200, -3), index=('y', 'x'), name='s')

        self.assertEqual(
            f1.insert_after('q', s1).to_pairs(),
            (
                ('p', (('x', 'a'), ('y', 'b'))),
                ('q', (('x', False), ('y', True))),
                ('s', (('x', -3), ('y', 200))),
                ('r', (('x', True), ('y', False))),
            ),
        )

        self.assertEqual(
            f1.insert_before('q', s1).to_pairs(),
            (
                ('p', (('x', 'a'), ('y', 'b'))),
                ('s', (('x', -3), ('y', 200))),
                ('q', (('x', False), ('y', True))),
                ('r', (('x', True), ('y', False))),
            ),
        )

    def test_frame_insert_f(self) -> None:
        records = (('a', False, True), ('b', True, False))
        f1 = FrameGO.from_records(records, columns=('p', 'q', 'r'), index=('x', 'y'))

        s1 = Series((200, -3), index=('y', 'x'), name='s')

        with self.assertRaises(RuntimeError):
            f1.insert_before(slice('q', 'r'), s1)

        with self.assertRaises(RuntimeError):
            f1.insert_after(slice('q', 'r'), s1)

    def test_frame_insert_g1(self) -> None:
        f = ff.parse('s(3,3)|v(str)')
        f = f.insert_after(
            sf.ILoc[-1], sf.Series.from_element(1, index=f.index, name='a')
        )

        self.assertEqual(
            f.to_pairs(),
            (
                (0, ((0, 'zjZQ'), (1, 'zO5l'), (2, 'zEdH'))),
                (1, ((0, 'zaji'), (1, 'zJnC'), (2, 'zDdR'))),
                (2, ((0, 'ztsv'), (1, 'zUvW'), (2, 'zkuW'))),
                ('a', ((0, 1), (1, 1), (2, 1))),
            ),
        )

        f = f.insert_after(
            sf.ILoc[-1], sf.Series.from_element(2, index=f.index, name='b')
        )
        self.assertEqual(
            f.to_pairs(),
            (
                (0, ((0, 'zjZQ'), (1, 'zO5l'), (2, 'zEdH'))),
                (1, ((0, 'zaji'), (1, 'zJnC'), (2, 'zDdR'))),
                (2, ((0, 'ztsv'), (1, 'zUvW'), (2, 'zkuW'))),
                ('a', ((0, 1), (1, 1), (2, 1))),
                ('b', ((0, 2), (1, 2), (2, 2))),
            ),
        )

    def test_frame_insert_g2(self) -> None:
        # a Frame with four columns, no rows; inserting an empty Series will work as with FrameGO
        empty = sf.Frame.from_records([], columns=list('abcd'))
        new_column = sf.Series(np.array(()), name='e')
        f2 = empty.insert_after(sf.ILoc[-1], new_column)
        self.assertEqual(f2.shape, (0, 5))
        self.assertEqual(f2.columns.values.tolist(), ['a', 'b', 'c', 'd', 'e'])

        # this works too as we take the intersection index
        new_column = sf.Series(
            [
                0,
            ],
            index=('a',),
            name='e',
        )
        f3 = empty.insert_after(sf.ILoc[-1], new_column)
        self.assertEqual(f3.shape, (0, 5))
        self.assertEqual(f3.columns.values.tolist(), ['a', 'b', 'c', 'd', 'e'])

    def test_frame_insert_h(self) -> None:
        f = ff.parse('s(2,3)|v(str)')
        f = f.insert_after(
            sf.ILoc[-2], sf.Series.from_element(1, index=f.index, name='a')
        )
        self.assertEqual(
            f.to_pairs(),
            (
                (0, ((0, 'zjZQ'), (1, 'zO5l'))),
                (1, ((0, 'zaji'), (1, 'zJnC'))),
                ('a', ((0, 1), (1, 1))),
                (2, ((0, 'ztsv'), (1, 'zUvW'))),
            ),
        )

    def test_frame_insert_i(self) -> None:
        f = ff.parse('s(2,3)|v(str)')
        f = f.insert_before(
            sf.ILoc[-2], sf.Series.from_element(1, index=f.index, name='a')
        )
        f = f.insert_before(
            sf.ILoc[-2], sf.Series.from_element(2, index=f.index, name='b')
        )
        self.assertEqual(
            f.to_pairs(),
            (
                (0, ((0, 'zjZQ'), (1, 'zO5l'))),
                ('a', ((0, 1), (1, 1))),
                ('b', ((0, 2), (1, 2))),
                (1, ((0, 'zaji'), (1, 'zJnC'))),
                (2, ((0, 'ztsv'), (1, 'zUvW'))),
            ),
        )

    def test_frame_insert_j(self) -> None:
        f = ff.parse('s(2,3)|v(str)')
        with self.assertRaises(IndexError):
            _ = f.insert_after(
                sf.ILoc[3], sf.Series.from_element(1, index=f.index, name='a')
            )
        with self.assertRaises(IndexError):
            _ = f.insert_before(
                sf.ILoc[3], sf.Series.from_element(1, index=f.index, name='a')
            )
        with self.assertRaises(IndexError):
            _ = f.insert_after(
                sf.ILoc[-4], sf.Series.from_element(1, index=f.index, name='a')
            )
        with self.assertRaises(IndexError):
            _ = f.insert_before(
                sf.ILoc[-4], sf.Series.from_element(1, index=f.index, name='a')
            )

    # ---------------------------------------------------------------------------

    def test_frame_extract_a(self) -> None:
        # reindex both axis
        records = (
            (2, 2, 'a', False, False),
            (30, 34, 'b', True, False),
            (2, 95, 'c', False, False),
            (30, 73, 'd', True, True),
        )

        f1 = Frame.from_records(
            records, columns=('p', 'q', 'r', 's', 't'), index=('w', 'x', 'y', 'z')
        )

        f2 = f1._extract(row_key=np.array((False, True, True, False), dtype=bool))

        self.assertEqual(
            f2.to_pairs(),
            (
                ('p', (('x', 30), ('y', 2))),
                ('q', (('x', 34), ('y', 95))),
                ('r', (('x', 'b'), ('y', 'c'))),
                ('s', (('x', True), ('y', False))),
                ('t', (('x', False), ('y', False))),
            ),
        )

        f3 = f1._extract(row_key=np.array((True, False, False, True), dtype=bool))

        self.assertEqual(
            f3.to_pairs(),
            (
                ('p', (('w', 2), ('z', 30))),
                ('q', (('w', 2), ('z', 73))),
                ('r', (('w', 'a'), ('z', 'd'))),
                ('s', (('w', False), ('z', True))),
                ('t', (('w', False), ('z', True))),
            ),
        )

        # attempting to select any single row results in a problem, as the first block given to the TypeBlocks constructor is a 1d array that looks it is a (2,1) instead of a (1, 2)
        f4 = f1._extract(row_key=np.array((False, False, True, False), dtype=bool))

        self.assertEqual(
            f4.to_pairs(),
            (
                ('p', (('y', 2),)),
                ('q', (('y', 95),)),
                ('r', (('y', 'c'),)),
                ('s', (('y', False),)),
                ('t', (('y', False),)),
            ),
        )

    def test_frame_extract_b(self) -> None:
        # examining cases where shape goes to zero in one dimension

        f1 = Frame.from_element(None, index=tuple('ab'), columns=('c',))
        f2: Frame = f1[[]]  # type: ignore
        self.assertEqual(len(f2.columns), 0)
        self.assertEqual(len(f2.index), 2)
        self.assertEqual(f2.shape, (2, 0))

    def test_frame_extract_c(self) -> None:
        # examining cases where shape goes to zero in one dimension
        f1 = Frame.from_element(None, columns=tuple('ab'), index=('c',))
        f2 = f1.loc[[]]
        self.assertEqual(f2.shape, (0, 2))
        self.assertEqual(len(f2.columns), 2)
        self.assertEqual(len(f2.index), 0)

    def test_frame_extract_d(self) -> None:
        # examining cases where shape goes to zero in one dimension
        f = sf.Frame.from_element(True, index=[1, 2, 3], columns=['a'])
        target = sf.Series([False, False, False], index=[1, 2, 3])

        self.assertEqual(f.loc[target, 'a'].dtype, np.dtype('bool'))  # type: ignore
        self.assertEqual(f.loc[target].dtypes.values.tolist(), [np.dtype('bool')])

    def test_frame_extract_e(self) -> None:
        # examining cases where shape goes to zero in one dimension
        f = sf.Frame.from_element('fourty-two', index=[1, 2, 3], columns=['a'])
        target = sf.Series([False, False, False], index=[1, 2, 3])

        self.assertEqual(f.loc[target, 'a'].dtype, np.dtype('<U10'))  # type: ignore
        self.assertEqual(f.loc[target].dtypes.values.tolist(), [np.dtype('<U10')])

    def test_frame_extract_f(self) -> None:
        # examining cases where shape goes to zero in one dimension

        self.assertEqual(
            sf.Frame.from_records(([3.1, None, 'foo'],)).loc[[], 0].dtype,  # type: ignore
            np.dtype('float64'),
        )
        self.assertEqual(
            sf.Frame.from_records(([3.1, None, 'foo'],)).loc[[], 1].dtype,  # type: ignore
            np.dtype('O'),
        )
        self.assertEqual(
            sf.Frame.from_records(([3.1, None, 'foo'],)).loc[[], 2].dtype,  # type: ignore
            np.dtype('<U3'),
        )

    # ---------------------------------------------------------------------------

    def test_frame_loc_a(self) -> None:
        records = ((1, 2, 'a', False, True), (30, 50, 'b', True, False))

        f1 = Frame.from_records(
            records, columns=('p', 'q', 'r', 's', 't'), index=('x', 'y')
        )

        # cases of single series extraction
        s1 = f1.loc['x']
        self.assertEqual(
            list(s1.items()),
            [('p', 1), ('q', 2), ('r', 'a'), ('s', False), ('t', True)],
        )

        s2 = f1.loc[:, 'p']
        self.assertEqual(list(s2.items()), [('x', 1), ('y', 30)])

        self.assertEqual(f1.loc[['y', 'x']].index.values.tolist(), ['y', 'x'])

        self.assertEqual(
            f1['r':].columns.values.tolist(),  # type: ignore  # https://github.com/python/typeshed/pull/3024
            ['r', 's', 't'],
        )

    def test_frame_loc_b(self) -> None:
        # dimensionality of returned item based on selectors
        records = ((1, 2, 'a', False, True), (30, 50, 'b', True, False))

        f1 = Frame.from_records(
            records, columns=('p', 'q', 'r', 's', 't'), index=('x', 'y')
        )

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
        self.assertEqual(post.index.values.tolist(), ['s', 't'])

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

        f1 = Frame.from_records(
            records, columns=('p', 'q', 'r', 's', 't'), index=('w', 'x', 'y', 'z')
        )

        post = f1.loc['x':]  # type: ignore  # https://github.com/python/typeshed/pull/3024
        self.assertEqual(post.index.values.tolist(), ['x', 'y', 'z'])

    def test_frame_loc_d(self) -> None:
        records = ((1, 2, 'a', False, True), (30, 50, 'b', True, False))

        f1 = Frame.from_records(
            records, columns=('p', 'q', 'r', 's', 't'), index=('x', 'y'), name='foo'
        )

        f2 = f1['r':]  # type: ignore  # https://github.com/python/typeshed/pull/3024
        f3 = f1.loc[['y'], ['r']]
        self.assertEqual(f1.name, 'foo')
        self.assertEqual(f2.name, 'foo')
        self.assertEqual(f3.name, 'foo')

        s1 = f2.loc[:, 's']
        self.assertEqual(s1.name, 's')

        s2 = f1.loc['x', :'r']  # type: ignore  # https://github.com/python/typeshed/pull/3024
        self.assertEqual(s2.name, 'x')

    def test_frame_loc_f(self) -> None:
        f = Frame.from_elements(range(3), index=sf.Index(tuple('abc'), name='index'))
        self.assertEqual(f.loc['b':].index.name, 'index')  # type: ignore

    def test_frame_loc_g(self) -> None:
        f = Frame.from_dict(dict(a=[None], b=[1]))
        self.assertEqual(f.shape, (1, 2))
        post = f.loc[f['a'] == True]
        self.assertEqual(post.shape, (0, 2))

    def test_frame_loc_h(self) -> None:
        f1 = Frame(index=('a', 'b', 'c'))
        s1 = f1.loc['b']
        self.assertEqual(s1.name, 'b')
        self.assertEqual(len(s1), 0)

        f2 = Frame(columns=('a', 'b', 'c'))
        s2 = f2['c']
        self.assertEqual(s2.name, 'c')
        self.assertEqual(len(s2), 0)

    def test_frame_loc_i(self) -> None:
        f1 = Frame(np.arange(16).reshape((4, 4)))

        self.assertEqual(
            f1.loc[:2, :2].to_pairs(),
            (
                (0, ((0, 0), (1, 4), (2, 8))),
                (1, ((0, 1), (1, 5), (2, 9))),
                (2, ((0, 2), (1, 6), (2, 10))),
            ),
        )

        self.assertEqual(
            f1.iloc[:2, :2].to_pairs(), ((0, ((0, 0), (1, 4))), (1, ((0, 1), (1, 5))))
        )

    def test_frame_loc_j(self) -> None:
        f = sf.Frame.from_records(
            (
                ('muon', 0.106, -1.0, 'lepton'),
                ('tau', 1.777, -1.0, 'lepton'),
                ('charm', 1.3, 0.666, 'quark'),
                ('strange', 0.1, -0.333, 'quark'),
            ),
            columns=('name', 'mass', 'charge', 'type'),
        )

        f = f.set_index_hierarchy(('type', 'name'), drop=True)

        post1 = f.loc[HLoc[:, ['muon', 'strange']]]
        self.assertEqual(
            post1.to_pairs(),
            (
                ('mass', ((('lepton', 'muon'), 0.106), (('quark', 'strange'), 0.1))),
                (
                    'charge',
                    ((('lepton', 'muon'), -1.0), (('quark', 'strange'), -0.333)),
                ),
            ),
        )

    def test_frame_loc_k(self) -> None:
        f = sf.Frame.from_records(
            (
                ('muon', 0.106, -1.0, 'lepton'),
                ('tau', 1.777, -1.0, 'lepton'),
                ('charm', 1.3, 0.666, 'quark'),
                ('strange', 0.1, -0.333, 'quark'),
            ),
            columns=('name', 'mass', 'charge', 'type'),
        )

        f = f.set_index_hierarchy(('type', 'name'), drop=True)

        post2 = f.loc[HLoc[:, f['mass'] > 1]]

        self.assertEqual(
            post2.to_pairs(),
            (
                ('mass', ((('lepton', 'tau'), 1.777), (('quark', 'charm'), 1.3))),
                ('charge', ((('lepton', 'tau'), -1.0), (('quark', 'charm'), 0.666))),
            ),
        )

    def test_frame_loc_l(self) -> None:
        f1 = Frame.from_records((('a', [1, 2]), ('b', [3, 4])))
        s1 = f1.loc[0]
        self.assertEqual(s1.values.tolist(), ['a', [1, 2]])

        f3 = Frame.from_records(
            (('a', [1, 2], [10, 20]), ('b', [3, 4], [30, 40])), consolidate_blocks=True
        )
        self.assertEqual(f3.shape, (2, 3))
        self.assertEqual(f3.consolidate.status['shape'].values.tolist(), [(2,), (2, 2)])

        s2 = f3.loc[1]
        self.assertEqual(s2.values.tolist(), ['b', [3, 4], [30, 40]])

    def test_frame_loc_m(self) -> None:
        f1 = Frame.from_records((('a', [1, 2], [10, 20]), ('b', [3, 4], [30, 40])))
        s1 = f1.loc[0, 1:]
        self.assertEqual(s1.values.tolist(), [[1, 2], [10, 20]])

    # ---------------------------------------------------------------------------

    def test_frame_items_a(self) -> None:
        records = ((1, 2, 'a', False, True), (30, 50, 'b', True, False))

        f1 = Frame.from_records(
            records, columns=('p', 'q', 'r', 's', 't'), index=('x', 'y')
        )

        self.assertEqual(
            list((k, list(v.items())) for k, v in f1.items()),
            [
                ('p', [('x', 1), ('y', 30)]),
                ('q', [('x', 2), ('y', 50)]),
                ('r', [('x', 'a'), ('y', 'b')]),
                ('s', [('x', False), ('y', True)]),
                ('t', [('x', True), ('y', False)]),
            ],
        )

    def test_frame_items_b(self) -> None:
        records = ((1, True), (30, False))

        f1 = Frame.from_records(records, columns=('p', 'q'), index=('x', 'y'))

        for label, series in f1.items():
            self.assertEqual(series.name, label)

    # ---------------------------------------------------------------------------

    @skip_win
    def test_frame_attrs_a(self) -> None:
        records = ((1, 2, 'a', False, True), (30, 50, 'b', True, False))

        f1 = Frame.from_records(
            records, columns=('p', 'q', 'r', 's', 't'), index=('x', 'y')
        )

        self.assertEqual(
            str(f1.dtypes.values.tolist()),
            "[dtype('int64'), dtype('int64'), dtype('<U1'), dtype('bool'), dtype('bool')]",
        )

        self.assertEqual(f1.size, 10)
        self.assertEqual(f1.ndim, 2)
        self.assertEqual(f1.shape, (2, 5))

    # ---------------------------------------------------------------------------

    def test_frame_assign_getitem_a(self) -> None:
        f1 = FrameGO(index=(0, 1, 2))
        for idx, col in enumerate(string.ascii_uppercase + string.ascii_lowercase):
            f1[col] = idx

        fields = [
            'm',
            'V',
            'P',
            'c',
            'Y',
            'r',
            'q',
            'R',
            'j',
            'X',
            'a',
            'E',
            'K',
            'p',
            'u',
            'G',
            'D',
            'w',
            'd',
            'e',
            'H',
            'i',
            'h',
            'N',
            'O',
            'k',
            'l',
            'F',
            'g',
            'o',
            'M',
            'T',
            'n',
            'L',
            'Q',
            'W',
            't',
            'v',
            's',
            'Z',
            'J',
            'I',
            'b',
        ]

        # check that normal selection works
        f1_sub: Frame = f1[fields]  # type: ignore
        self.assertEqual(f1_sub.columns.values.tolist(), fields)

        f2 = f1.assign[fields](f1[fields] * 0.5)

        self.assertTrue((f2.columns == f1.columns).all())
        self.assertTrue((f2.index == f1.index).all())

        # as expected, values is coercing all to floats
        self.assertEqual(
            f2.values.tolist(),
            [
                [
                    0.0,
                    1.0,
                    2.0,
                    1.5,
                    2.0,
                    2.5,
                    3.0,
                    3.5,
                    4.0,
                    4.5,
                    5.0,
                    5.5,
                    6.0,
                    6.5,
                    7.0,
                    7.5,
                    8.0,
                    8.5,
                    18.0,
                    9.5,
                    20.0,
                    10.5,
                    11.0,
                    11.5,
                    12.0,
                    12.5,
                    13.0,
                    13.5,
                    14.0,
                    14.5,
                    15.0,
                    31.0,
                    16.0,
                    16.5,
                    17.0,
                    17.5,
                    18.0,
                    18.5,
                    19.0,
                    19.5,
                    20.0,
                    20.5,
                    21.0,
                    21.5,
                    22.0,
                    22.5,
                    23.0,
                    23.5,
                    24.0,
                    49.0,
                    50.0,
                    51.0,
                ],
                [
                    0.0,
                    1.0,
                    2.0,
                    1.5,
                    2.0,
                    2.5,
                    3.0,
                    3.5,
                    4.0,
                    4.5,
                    5.0,
                    5.5,
                    6.0,
                    6.5,
                    7.0,
                    7.5,
                    8.0,
                    8.5,
                    18.0,
                    9.5,
                    20.0,
                    10.5,
                    11.0,
                    11.5,
                    12.0,
                    12.5,
                    13.0,
                    13.5,
                    14.0,
                    14.5,
                    15.0,
                    31.0,
                    16.0,
                    16.5,
                    17.0,
                    17.5,
                    18.0,
                    18.5,
                    19.0,
                    19.5,
                    20.0,
                    20.5,
                    21.0,
                    21.5,
                    22.0,
                    22.5,
                    23.0,
                    23.5,
                    24.0,
                    49.0,
                    50.0,
                    51.0,
                ],
                [
                    0.0,
                    1.0,
                    2.0,
                    1.5,
                    2.0,
                    2.5,
                    3.0,
                    3.5,
                    4.0,
                    4.5,
                    5.0,
                    5.5,
                    6.0,
                    6.5,
                    7.0,
                    7.5,
                    8.0,
                    8.5,
                    18.0,
                    9.5,
                    20.0,
                    10.5,
                    11.0,
                    11.5,
                    12.0,
                    12.5,
                    13.0,
                    13.5,
                    14.0,
                    14.5,
                    15.0,
                    31.0,
                    16.0,
                    16.5,
                    17.0,
                    17.5,
                    18.0,
                    18.5,
                    19.0,
                    19.5,
                    20.0,
                    20.5,
                    21.0,
                    21.5,
                    22.0,
                    22.5,
                    23.0,
                    23.5,
                    24.0,
                    49.0,
                    50.0,
                    51.0,
                ],
            ],
        )

    def test_frame_assign_getitem_b(self) -> None:
        f1 = FrameGO(index=(0, 1, 2))
        for idx, col in enumerate('abcdef'):
            f1[col] = idx

        f2 = f1.assign['b'](Series(('b', 'c', 'd'), index=(2, 0, 1)))
        self.assertEqual(f2['b'].to_pairs(), ((0, 'c'), (1, 'd'), (2, 'b')))

    def test_frame_assign_getitem_c(self) -> None:
        f1 = sf.Frame.from_element(False, index=range(2), columns=tuple('ab'))
        f2 = f1.assign['a']([1.1, 2.1])
        self.assertEqual(f2._blocks.shapes.tolist(), [(2,), (2, 1)])
        self.assertEqual(
            f2.dtypes.values.tolist(), [np.dtype('float64'), np.dtype('bool')]
        )

        self.assertEqual(
            f2.to_pairs(),
            (('a', ((0, 1.1), (1, 2.1))), ('b', ((0, False), (1, False)))),
        )

    def test_frame_assign_getitem_d(self) -> None:
        f1 = sf.Frame.from_element(False, index=range(2), columns=tuple('abcd'))
        f2 = f1.assign['b']([1.1, 2.1])
        self.assertEqual(f2._blocks.shapes.tolist(), [(2, 1), (2,), (2, 2)])
        self.assertEqual(
            f2.dtypes.values.tolist(),
            [np.dtype('bool'), np.dtype('float64'), np.dtype('bool'), np.dtype('bool')],
        )
        self.assertEqual(
            f2.to_pairs(),
            (
                ('a', ((0, False), (1, False))),
                ('b', ((0, 1.1), (1, 2.1))),
                ('c', ((0, False), (1, False))),
                ('d', ((0, False), (1, False))),
            ),
        )

    def test_frame_assign_getitem_e(self) -> None:
        f1 = sf.Frame.from_element(False, index=range(2), columns=tuple('abcd'))
        f2 = f1.assign['c']([1.1, 2.1])
        self.assertEqual(f2._blocks.shapes.tolist(), [(2, 2), (2,), (2, 1)])
        self.assertEqual(
            f2.dtypes.values.tolist(),
            [np.dtype('bool'), np.dtype('bool'), np.dtype('float64'), np.dtype('bool')],
        )
        self.assertEqual(
            f2.to_pairs(),
            (
                ('a', ((0, False), (1, False))),
                ('b', ((0, False), (1, False))),
                ('c', ((0, 1.1), (1, 2.1))),
                ('d', ((0, False), (1, False))),
            ),
        )

    def test_frame_assign_getitem_f(self) -> None:
        f1 = sf.Frame.from_element(False, index=range(2), columns=tuple('abcd'))
        f2 = f1.assign['d']([1.1, 2.1])
        self.assertEqual(
            f2._blocks.shapes.tolist(),
            [
                (2, 3),
                (2,),
            ],
        )
        self.assertEqual(
            f2.dtypes.values.tolist(),
            [np.dtype('bool'), np.dtype('bool'), np.dtype('bool'), np.dtype('float64')],
        )
        self.assertEqual(
            f2.to_pairs(),
            (
                ('a', ((0, False), (1, False))),
                ('b', ((0, False), (1, False))),
                ('c', ((0, False), (1, False))),
                ('d', ((0, 1.1), (1, 2.1))),
            ),
        )

    def test_frame_assign_getitem_g(self) -> None:
        f1 = sf.Frame.from_element(False, index=range(2), columns=tuple('abcde'))
        f2 = f1.assign['b':'d']('x')  # type: ignore
        self.assertEqual(f2._blocks.shapes.tolist(), [(2, 1), (2, 3), (2, 1)])
        self.assertEqual(
            f2.dtypes.values.tolist(),
            [
                np.dtype('bool'),
                np.dtype('<U1'),
                np.dtype('<U1'),
                np.dtype('<U1'),
                np.dtype('bool'),
            ],
        )
        self.assertEqual(
            f2.to_pairs(),
            (
                ('a', ((0, False), (1, False))),
                ('b', ((0, 'x'), (1, 'x'))),
                ('c', ((0, 'x'), (1, 'x'))),
                ('d', ((0, 'x'), (1, 'x'))),
                ('e', ((0, False), (1, False))),
            ),
        )

    def test_frame_assign_getitem_h(self) -> None:
        f1 = sf.Frame.from_element('1', index=range(4), columns=range(5))
        f2 = f1[[1, 2, 3]].astype(
            {
                1: int,
                2: bool,
            }
        )
        f3 = f1.assign[f2.columns](f2)  # type: ignore
        self.assertEqual(
            [b.dtype.kind for b in f3._blocks._blocks], ['U', 'i', 'b', 'U', 'U']
        )
        self.assertEqual(
            f3.to_pairs(),
            (
                (0, ((0, '1'), (1, '1'), (2, '1'), (3, '1'))),
                (1, ((0, 1), (1, 1), (2, 1), (3, 1))),
                (2, ((0, True), (1, True), (2, True), (3, True))),
                (3, ((0, '1'), (1, '1'), (2, '1'), (3, '1'))),
                (4, ((0, '1'), (1, '1'), (2, '1'), (3, '1'))),
            ),
        )

    def test_frame_assign_getitem_i(self) -> None:
        f1 = ff.parse('s(2,10)|v(int)')
        f2 = f1.assign.loc[:, f1.columns % 2 == 0](0)

        self.assertEqual(
            f2.to_pairs(),
            (
                (0, ((0, 0), (1, 0))),
                (1, ((0, 162197), (1, -41157))),
                (2, ((0, 0), (1, 0))),
                (3, ((0, 129017), (1, 35021))),
                (4, ((0, 0), (1, 0))),
                (5, ((0, 84967), (1, 13448))),
                (6, ((0, 0), (1, 0))),
                (7, ((0, 137759), (1, -62964))),
                (8, ((0, 0), (1, 0))),
                (9, ((0, 126025), (1, 59728))),
            ),
        )

        self.assertEqual(
            f1.loc[:, f1.columns % 2 == 0].columns.values.tolist(), [0, 2, 4, 6, 8]
        )

    # ---------------------------------------------------------------------------

    def test_frame_assign_iloc_a(self) -> None:
        records = ((1, 2, 'a', False, True), (30, 50, 'b', True, False))

        f1 = Frame.from_records(
            records, columns=('p', 'q', 'r', 's', 't'), index=('x', 'y')
        )

        self.assertEqual(f1.assign.iloc[1, 1](3000).iloc[1, 1], 3000)

    def test_frame_assign_iloc_b(self) -> None:
        f1 = Frame.from_fields(
            (
                (10, 2, 8, 3),
                (False, True, True, False),
                ('1517-01-01', '1517-04-01', '1517-12-31', '1517-06-30'),
            ),
            columns=('a', 'b', 'c'),
            dtypes=dict(c=np.datetime64),
            name='x',
        )
        f2 = f1.assign.iloc[2]([-1, False, np.datetime64('2022-01-10')])
        self.assertEqual(f1.dtypes.values.tolist(), f2.dtypes.values.tolist())

        self.assertEqual(
            f2.to_pairs(),
            (
                ('a', ((0, 10), (1, 2), (2, -1), (3, 3))),
                ('b', ((0, False), (1, True), (2, False), (3, False))),
                (
                    'c',
                    (
                        (0, np.datetime64('1517-01-01')),
                        (1, np.datetime64('1517-04-01')),
                        (2, np.datetime64('2022-01-10')),
                        (3, np.datetime64('1517-06-30')),
                    ),
                ),
            ),
        )

    def test_frame_assign_iloc_c(self) -> None:
        f1 = Frame.from_fields(
            (
                (10, 2, 8, 3),
                (False, True, True, False),
                ('1517-01-01', '1517-04-01', '1517-12-31', '1517-06-30'),
            ),
            columns=('a', 'b', 'c'),
            dtypes=dict(c=np.datetime64),
            name='x',
        )
        f2 = f1.assign.iloc[2, [0, 2]]([-1, np.datetime64('2022-01-10')])
        self.assertEqual(f1.dtypes.values.tolist(), f2.dtypes.values.tolist())
        self.assertEqual(
            f2.to_pairs(),
            (
                ('a', ((0, 10), (1, 2), (2, -1), (3, 3))),
                ('b', ((0, False), (1, True), (2, True), (3, False))),
                (
                    'c',
                    (
                        (0, np.datetime64('1517-01-01')),
                        (1, np.datetime64('1517-04-01')),
                        (2, np.datetime64('2022-01-10')),
                        (3, np.datetime64('1517-06-30')),
                    ),
                ),
            ),
        )

    def test_frame_assign_iloc_d(self) -> None:
        f1 = Frame.from_fields(
            (
                (10, 2, 8, 3),
                (10, 2, 8, 3),
                (False, True, True, False),
                (False, True, True, False),
                ('1517-01-01', '1517-04-01', '1517-12-31', '1517-06-30'),
            ),
            columns=('a', 'b', 'c', 'd', 'e'),
            dtypes=dict(e=np.datetime64),
            consolidate_blocks=True,
        )
        f2 = f1.assign.iloc[2, 2:]([False, False, np.datetime64('2022-01-10')])
        self.assertEqual(f1.dtypes.values.tolist(), f2.dtypes.values.tolist())
        self.assertEqual(
            f2.to_pairs(),
            (
                ('a', ((0, 10), (1, 2), (2, 8), (3, 3))),
                ('b', ((0, 10), (1, 2), (2, 8), (3, 3))),
                ('c', ((0, False), (1, True), (2, False), (3, False))),
                ('d', ((0, False), (1, True), (2, False), (3, False))),
                (
                    'e',
                    (
                        (0, np.datetime64('1517-01-01')),
                        (1, np.datetime64('1517-04-01')),
                        (2, np.datetime64('2022-01-10')),
                        (3, np.datetime64('1517-06-30')),
                    ),
                ),
            ),
        )

    def test_frame_assign_iloc_e(self) -> None:
        f1 = Frame.from_fields(
            (
                (10, 2, 8, 3),
                (10, 2, 8, 3),
                (False, True, True, False),
                (False, True, True, False),
                ('1517-01-01', '1517-04-01', '1517-12-31', '1517-06-30'),
            ),
            columns=('a', 'b', 'c', 'd', 'e'),
            dtypes=dict(e=np.datetime64),
            consolidate_blocks=True,
        )
        f2 = f1.assign.iloc[1:, 3:]([False, np.datetime64('2022-01-10')])
        self.assertEqual(f1.dtypes.values.tolist(), f2.dtypes.values.tolist())
        self.assertEqual(
            f2.to_pairs(),
            (
                ('a', ((0, 10), (1, 2), (2, 8), (3, 3))),
                ('b', ((0, 10), (1, 2), (2, 8), (3, 3))),
                ('c', ((0, False), (1, True), (2, True), (3, False))),
                ('d', ((0, False), (1, False), (2, False), (3, False))),
                (
                    'e',
                    (
                        (0, np.datetime64('1517-01-01')),
                        (1, np.datetime64('2022-01-10')),
                        (2, np.datetime64('2022-01-10')),
                        (3, np.datetime64('2022-01-10')),
                    ),
                ),
            ),
        )

    def test_frame_assign_iloc_f(self) -> None:
        f1 = Frame.from_fields(
            (
                (10, 2, 8),
                (False, False, False),
                (False, False, False),
                ('1517-01-01', '1517-04-01', '1517-12-31'),
            ),
            columns=('a', 'b', 'c', 'd'),
            dtypes=dict(d=np.datetime64),
            consolidate_blocks=True,
        )
        f2 = f1.assign.iloc[0, 0]([100])
        self.assertEqual(f1.dtypes.values.tolist(), f2.dtypes.values.tolist())
        self.assertEqual(
            f2.to_pairs(),
            (
                ('a', ((0, 100), (1, 2), (2, 8))),
                ('b', ((0, False), (1, False), (2, False))),
                ('c', ((0, False), (1, False), (2, False))),
                (
                    'd',
                    (
                        (0, np.datetime64('1517-01-01')),
                        (1, np.datetime64('1517-04-01')),
                        (2, np.datetime64('1517-12-31')),
                    ),
                ),
            ),
        )

    def test_frame_assign_iloc_h(self) -> None:
        f1 = Frame.from_fields(
            (
                (10, 2, 8),
                (False, False, False),
                (False, False, False),
                ('1517-01-01', '1517-04-01', '1517-12-31'),
            ),
            columns=('a', 'b', 'c', 'd'),
            dtypes=dict(d=np.datetime64),
            consolidate_blocks=True,
        )
        f2 = f1.assign.iloc[0, 0](())
        # self.assertEqual(f1.dtypes.values.tolist(), f2.dtypes.values.tolist())
        self.assertEqual(
            [dt.kind for dt in f2.dtypes.values],
            ['O', 'b', 'b', 'M'],
        )
        self.assertEqual(f2['a'].values.tolist(), [(), 2, 8])

    # ---------------------------------------------------------------------------

    def test_frame_assign_loc_a(self) -> None:
        records = ((1, 2, 'a', False, True), (30, 50, 'b', True, False))

        f1 = Frame.from_records(
            records, columns=('p', 'q', 'r', 's', 't'), index=('x', 'y')
        )

        self.assertEqual(
            f1.assign.loc['x', 's'](3000).values.tolist(),
            [[1, 2, 'a', 3000, True], [30, 50, 'b', True, False]],
        )

        # can assign to a columne
        self.assertEqual(
            f1.assign['s']('x').values.tolist(),
            [[1, 2, 'a', 'x', True], [30, 50, 'b', 'x', False]],
        )

    def test_frame_assign_loc_b(self) -> None:
        records = ((1, 2, 'a', False, True), (30, 50, 'b', True, False))

        f1 = Frame.from_records(
            records, columns=('p', 'q', 'r', 's', 't'), index=('x', 'y')
        )

        # unindexed tuple/list assingment
        self.assertEqual(
            f1.assign['s']([50, 40]).values.tolist(),
            [[1, 2, 'a', 50, True], [30, 50, 'b', 40, False]],
        )

        self.assertEqual(
            f1.assign.loc['y'](list(range(5))).values.tolist(),
            [[1, 2, 'a', False, True], [0, 1, 2, 3, 4]],
        )

    def test_frame_assign_loc_c(self) -> None:
        records = ((1, 2, 'a', False, True), (30, 50, 'b', True, False))

        f1 = Frame.from_records(
            records, columns=('p', 'q', 'r', 's', 't'), index=('x', 'y')
        )

        # assinging a series to a part of wone row
        post = f1.assign.loc['x', 'r':](Series((-1, -2, -3), index=('t', 'r', 's')))  # type: ignore  # https://github.com/python/typeshed/pull/3024

        self.assertEqual(
            post.values.tolist(), [[1, 2, -2, -3, -1], [30, 50, 'b', True, False]]
        )

        post = f1.assign.loc[['x', 'y'], 'r'](Series((-1, -2), index=('y', 'x')))

        self.assertEqual(
            post.values.tolist(), [[1, 2, -2, False, True], [30, 50, -1, True, False]]
        )

        # ordere here does not matter
        post = f1.assign.loc[['y', 'x'], 'r'](Series((-1, -2), index=('y', 'x')))

        self.assertEqual(
            post.values.tolist(), [[1, 2, -2, False, True], [30, 50, -1, True, False]]
        )

    def test_frame_assign_loc_d(self) -> None:
        records = ((1, 2, 'a', False, True), (30, 50, 'b', True, False))

        f1 = Frame.from_records(
            records,
            columns=('p', 'q', 'r', 's', 't'),
            index=('x', 'y'),
            consolidate_blocks=True,
        )

        value1 = Frame.from_records(
            ((20, 21, 22), (23, 24, 25)),
            index=('x', 'y'),
            columns=('s', 't', 'w'),
            consolidate_blocks=True,
        )

        f2 = f1.assign.loc[['x', 'y'], ['s', 't']](value1)
        self.assertEqual(
            f2.to_pairs(),
            (
                ('p', (('x', 1), ('y', 30))),
                ('q', (('x', 2), ('y', 50))),
                ('r', (('x', 'a'), ('y', 'b'))),
                ('s', (('x', 20), ('y', 23))),
                ('t', (('x', 21), ('y', 24))),
            ),
        )

    def test_frame_assign_loc_e(self) -> None:
        records = (
            (1, 2, 'a', False, True),
            (30, 50, 'b', True, False),
            (30, 50, 'c', False, False),
            (3, 5, 'c', False, True),
            (30, 500, 'd', True, True),
            (30, 2, 'e', True, True),
        )

        f1 = Frame.from_records(
            records, columns=('p', 'q', 'r', 's', 't'), index=list('abcdef')
        )

        f3 = f1.assign.iloc[5](f1.iloc[0])
        self.assertEqual(
            f3.to_pairs(),
            (
                ('p', (('a', 1), ('b', 30), ('c', 30), ('d', 3), ('e', 30), ('f', 1))),
                ('q', (('a', 2), ('b', 50), ('c', 50), ('d', 5), ('e', 500), ('f', 2))),
                (
                    'r',
                    (
                        ('a', 'a'),
                        ('b', 'b'),
                        ('c', 'c'),
                        ('d', 'c'),
                        ('e', 'd'),
                        ('f', 'a'),
                    ),
                ),
                (
                    's',
                    (
                        ('a', False),
                        ('b', True),
                        ('c', False),
                        ('d', False),
                        ('e', True),
                        ('f', False),
                    ),
                ),
                (
                    't',
                    (
                        ('a', True),
                        ('b', False),
                        ('c', False),
                        ('d', True),
                        ('e', True),
                        ('f', True),
                    ),
                ),
            ),
        )

        f4 = f1.assign['q'](f1['q'] * 2)
        self.assertEqual(
            f4.to_pairs(),
            (
                ('p', (('a', 1), ('b', 30), ('c', 30), ('d', 3), ('e', 30), ('f', 30))),
                (
                    'q',
                    (
                        ('a', 4),
                        ('b', 100),
                        ('c', 100),
                        ('d', 10),
                        ('e', 1000),
                        ('f', 4),
                    ),
                ),
                (
                    'r',
                    (
                        ('a', 'a'),
                        ('b', 'b'),
                        ('c', 'c'),
                        ('d', 'c'),
                        ('e', 'd'),
                        ('f', 'e'),
                    ),
                ),
                (
                    's',
                    (
                        ('a', False),
                        ('b', True),
                        ('c', False),
                        ('d', False),
                        ('e', True),
                        ('f', True),
                    ),
                ),
                (
                    't',
                    (
                        ('a', True),
                        ('b', False),
                        ('c', False),
                        ('d', True),
                        ('e', True),
                        ('f', True),
                    ),
                ),
            ),
        )

        f5 = f1.assign[['p', 'q']](f1[['p', 'q']] * 2)
        self.assertEqual(
            f5.to_pairs(),
            (
                ('p', (('a', 2), ('b', 60), ('c', 60), ('d', 6), ('e', 60), ('f', 60))),
                (
                    'q',
                    (
                        ('a', 4),
                        ('b', 100),
                        ('c', 100),
                        ('d', 10),
                        ('e', 1000),
                        ('f', 4),
                    ),
                ),
                (
                    'r',
                    (
                        ('a', 'a'),
                        ('b', 'b'),
                        ('c', 'c'),
                        ('d', 'c'),
                        ('e', 'd'),
                        ('f', 'e'),
                    ),
                ),
                (
                    's',
                    (
                        ('a', False),
                        ('b', True),
                        ('c', False),
                        ('d', False),
                        ('e', True),
                        ('f', True),
                    ),
                ),
                (
                    't',
                    (
                        ('a', True),
                        ('b', False),
                        ('c', False),
                        ('d', True),
                        ('e', True),
                        ('f', True),
                    ),
                ),
            ),
        )

    def test_frame_assign_loc_f(self) -> None:
        f1 = sf.Frame.from_element(False, index=range(2), columns=tuple('abcde'))
        f2 = f1.assign.loc[1, 'b':'d']('x')  # type: ignore
        self.assertEqual(
            f2.dtypes.values.tolist(),
            [
                np.dtype('bool'),
                np.dtype('O'),
                np.dtype('O'),
                np.dtype('O'),
                np.dtype('bool'),
            ],
        )
        self.assertEqual(
            f2.to_pairs(),
            (
                ('a', ((0, False), (1, False))),
                ('b', ((0, False), (1, 'x'))),
                ('c', ((0, False), (1, 'x'))),
                ('d', ((0, False), (1, 'x'))),
                ('e', ((0, False), (1, False))),
            ),
        )

    def test_frame_assign_loc_g(self) -> None:
        f1 = ff.parse('s(4,8)|v(bool,bool,int,bool,int)')
        f2 = ff.parse('s(6,4)|v(str,dtY)').relabel(columns=(2, 5, 6, 8))

        self.assertEqual(
            f1.assign.loc[[2, 3], :](f2, fill_value=0).to_pairs(),
            (
                (0, ((0, False), (1, False), (2, 0), (3, 0))),
                (1, ((0, False), (1, False), (2, 0), (3, 0))),
                (2, ((0, -3648), (1, 91301), (2, 'zEdH'), (3, 'zB7E'))),
                (3, ((0, False), (1, False), (2, 0), (3, 0))),
                (4, ((0, 58768), (1, 146284), (2, 0), (3, 0))),
                (
                    5,
                    (
                        (0, False),
                        (1, True),
                        (2, datetime.date(7699, 1, 1)),
                        (3, 168387),
                    ),
                ),
                (6, ((0, True), (1, True), (2, 'zkuW'), (3, 'zmVj'))),
                (7, ((0, 137759), (1, -62964), (2, 0), (3, 0))),
            ),
        )

        self.assertEqual(
            f1.assign.loc[:, [2, 5, 6]](f2).to_pairs(),
            (
                (0, ((0, False), (1, False), (2, False), (3, True))),
                (1, ((0, False), (1, False), (2, False), (3, False))),
                (2, ((0, 'zjZQ'), (1, 'zO5l'), (2, 'zEdH'), (3, 'zB7E'))),
                (3, ((0, False), (1, False), (2, True), (3, True))),
                (4, ((0, 58768), (1, 146284), (2, 170440), (3, 32395))),
                (
                    5,
                    (
                        (0, np.datetime64('164167')),
                        (1, np.datetime64('43127')),
                        (2, np.datetime64('7699')),
                        (3, np.datetime64('170357')),
                    ),
                ),
                (6, ((0, 'ztsv'), (1, 'zUvW'), (2, 'zkuW'), (3, 'zmVj'))),
                (7, ((0, 137759), (1, -62964), (2, 172142), (3, -154686))),
            ),
        )

    def test_frame_assign_loc_h(self) -> None:
        f1 = ff.parse('s(3,4)|c(I,str)|v(int)')
        f2 = f1.assign.loc[1:, ['ztsv', 'zUvW']].apply(lambda f: -f)

        self.assertEqual(
            f2.to_pairs(),
            (
                ('zZbu', ((0, -88017), (1, 92867), (2, 84967))),
                ('ztsv', ((0, 162197), (1, 41157), (2, -5729))),
                ('zUvW', ((0, -3648), (1, -91301), (2, -30205))),
                ('zkuW', ((0, 129017), (1, 35021), (2, 166924))),
            ),
        )

    def test_frame_assign_loc_i(self) -> None:
        f1 = ff.parse('s(3,4)|c(I,str)')
        f2 = f1.assign.loc[[0, 2], ['zkuW', 'zUvW']](0)
        self.assertEqual(
            f2.to_pairs(),
            (
                ('zZbu', ((0, 1930.4), (1, -1760.34), (2, 1857.34))),
                ('ztsv', ((0, -610.8), (1, 3243.94), (2, -823.14))),
                ('zUvW', ((0, 0.0), (1, -72.96), (2, 0.0))),
                ('zkuW', ((0, 0.0), (1, 2580.34), (2, 0.0))),
            ),
        )

    def test_frame_assign_loc_j(self) -> None:
        f1 = sf.Frame.from_dict(
            dict(
                comp_id=(1, 2, 3, 1, 2, 3, 1, 2, 3),
                year=(2000, 2000, 2000, 2001, 2001, 2001, 2002, 2002, 2002),
                return_total=(
                    0.01,
                    -0.02,
                    0.05,
                    -0.015,
                    0.02,
                    0.015,
                    0.002,
                    0.003,
                    0.01,
                ),
                return_price=(
                    0.01,
                    -0.02,
                    0.05,
                    -0.015,
                    0.02,
                    0.015,
                    0.002,
                    0.003,
                    0.01,
                ),
            )
        )
        na_flds = ['return_price', 'return_total']
        na_cells = f1['comp_id'].isin((2,)) & f1['year'].isin((2001, 2002))

        f2 = f1.assign.loc[na_cells, na_flds](None)
        self.assertEqual(
            f2.to_pairs(),
            (
                (
                    'comp_id',
                    (
                        (0, 1),
                        (1, 2),
                        (2, 3),
                        (3, 1),
                        (4, 2),
                        (5, 3),
                        (6, 1),
                        (7, 2),
                        (8, 3),
                    ),
                ),
                (
                    'year',
                    (
                        (0, 2000),
                        (1, 2000),
                        (2, 2000),
                        (3, 2001),
                        (4, 2001),
                        (5, 2001),
                        (6, 2002),
                        (7, 2002),
                        (8, 2002),
                    ),
                ),
                (
                    'return_total',
                    (
                        (0, 0.01),
                        (1, -0.02),
                        (2, 0.05),
                        (3, -0.015),
                        (4, None),
                        (5, 0.015),
                        (6, 0.002),
                        (7, None),
                        (8, 0.01),
                    ),
                ),
                (
                    'return_price',
                    (
                        (0, 0.01),
                        (1, -0.02),
                        (2, 0.05),
                        (3, -0.015),
                        (4, None),
                        (5, 0.015),
                        (6, 0.002),
                        (7, None),
                        (8, 0.01),
                    ),
                ),
            ),
        )

    def test_frame_assign_loc_k(self) -> None:
        f1 = ff.parse('s(2,6)|c(I,int)|v(int)').relabel(columns=range(6))
        f2 = f1.assign.loc[1, [4, 2, 0, 1]](0)
        self.assertEqual(
            f2.to_pairs(),
            (
                (0, ((0, -88017), (1, 0))),
                (1, ((0, 162197), (1, 0))),
                (2, ((0, -3648), (1, 0))),
                (3, ((0, 129017), (1, 35021))),
                (4, ((0, 58768), (1, 0))),
                (5, ((0, 84967), (1, 13448))),
            ),
        )

        f3 = f1.assign.loc[1, [4, 2, 0, 1]](Series(tuple('abcd'), index=(0, 1, 2, 4)))
        self.assertEqual(
            f3.to_pairs(),
            (
                (0, ((0, -88017), (1, 'a'))),
                (1, ((0, 162197), (1, 'b'))),
                (2, ((0, -3648), (1, 'c'))),
                (3, ((0, 129017), (1, 35021))),
                (4, ((0, 58768), (1, 'd'))),
                (5, ((0, 84967), (1, 13448))),
            ),
        )

    def test_frame_assign_loc_l(self) -> None:
        f1 = ff.parse('s(2,4)|c(I,int)|v(int)').relabel(columns=range(4))
        f2 = f1.assign.loc[1](Series(('a', 'b'), index=(2, 1)), fill_value='')
        self.assertEqual(
            f2.to_pairs(),
            (
                (0, ((0, -88017), (1, ''))),
                (1, ((0, 162197), (1, 'b'))),
                (2, ((0, -3648), (1, 'a'))),
                (3, ((0, 129017), (1, ''))),
            ),
        )

    def test_frame_assign_loc_m1(self) -> None:
        f1 = ff.parse('s(2,4)|c(I,int)|v(int)').relabel(columns=range(4))
        f2 = f1.assign.loc[1, 2]((3, -1))
        self.assertEqual(f2.loc[1, 2], (3, -1))

    def test_frame_assign_loc_m2(self) -> None:
        f1 = ff.parse('s(2,4)|c(I,int)|v(int)').relabel(columns=range(4))
        f2 = f1.assign.loc[1, 2:]((3, -1, 5))
        self.assertEqual(
            f2.to_pairs(),
            (
                (0, ((0, -88017), (1, 92867))),
                (1, ((0, 162197), (1, -41157))),
                (2, ((0, -3648), (1, (3, -1, 5)))),
                (3, ((0, 129017), (1, (3, -1, 5)))),
            ),
        )

    def test_frame_assign_loc_m3(self) -> None:
        f1 = ff.parse('s(2,4)|c(I,int)|v(int)').relabel(columns=range(4))
        f2 = f1.assign.loc[1, 2:]((-1, 5))
        self.assertEqual(
            f2.to_pairs(),
            (
                (0, ((0, -88017), (1, 92867))),
                (1, ((0, 162197), (1, -41157))),
                (2, ((0, -3648), (1, (-1, 5)))),
                (3, ((0, 129017), (1, (-1, 5)))),
            ),
        )

    # ---------------------------------------------------------------------------

    def test_frame_assign_coercion_a(self) -> None:
        records = ((1, 2, 'a', False, True), (30, 50, 'b', True, False))

        f1 = Frame.from_records(
            records, columns=('p', 'q', 'r', 's', 't'), index=('x', 'y')
        )
        f2 = f1.assign.loc['x', 'r'](None)
        self.assertEqual(
            f2.to_pairs(),
            (
                ('p', (('x', 1), ('y', 30))),
                ('q', (('x', 2), ('y', 50))),
                ('r', (('x', None), ('y', 'b'))),
                ('s', (('x', False), ('y', True))),
                ('t', (('x', True), ('y', False))),
            ),
        )

    # ---------------------------------------------------------------------------

    def test_frame_assign_bloc_a(self) -> None:
        records = ((1, 2, 'a', False, True), (30, 50, 'b', True, False))

        f1 = Frame.from_records(
            records, columns=('p', 'q', 'r', 's', 't'), index=('x', 'y')
        )

        sel = np.array(
            [[False, True, False, True, False], [True, False, True, False, True]]
        )
        f2 = f1.assign.bloc[sel](-10)
        self.assertEqual(
            f2.to_pairs(),
            (
                ('p', (('x', 1), ('y', -10))),
                ('q', (('x', -10), ('y', 50))),
                ('r', (('x', 'a'), ('y', -10))),
                ('s', (('x', -10), ('y', True))),
                ('t', (('x', True), ('y', -10))),
            ),
        )

        f3 = f1.assign.bloc[sel](None)
        self.assertEqual(
            f3.to_pairs(),
            (
                ('p', (('x', 1), ('y', None))),
                ('q', (('x', None), ('y', 50))),
                ('r', (('x', 'a'), ('y', None))),
                ('s', (('x', None), ('y', True))),
                ('t', (('x', True), ('y', None))),
            ),
        )

    def test_frame_assign_bloc_b(self) -> None:
        records = ((1, 2, 20, 40), (30, 50, -4, 5))

        f1 = Frame.from_records(
            records,
            columns=(
                'p',
                'q',
                'r',
                's',
            ),
            index=('x', 'y'),
        )

        sel = np.array([[False, True, False, True], [True, False, True, False]])

        # assignment from a frame of the same size
        f2 = f1.assign.bloc[sel](f1 * 100)

        self.assertEqual(
            f2.to_pairs(),
            (
                ('p', (('x', 1), ('y', 3000))),
                ('q', (('x', 200), ('y', 50))),
                ('r', (('x', 20), ('y', -400))),
                ('s', (('x', 4000), ('y', 5))),
            ),
        )

    def test_frame_assign_bloc_c(self) -> None:
        records = ((1, 2, 20, 40), (30, 50, -4, 5))

        f1 = Frame.from_records(
            records,
            columns=(
                'p',
                'q',
                'r',
                's',
            ),
            index=('x', 'y'),
        )

        sel = f1 >= 20

        f2 = f1.assign.bloc[sel](f1 * 100)

        match = (
            ('p', (('x', 1), ('y', 3000))),
            ('q', (('x', 2), ('y', 5000))),
            ('r', (('x', 2000), ('y', -4))),
            ('s', (('x', 4000), ('y', 5))),
        )

        self.assertEqual(f2.to_pairs(), match)

        # reording the value will have no affect
        f3 = f1.reindex(columns=('r', 'q', 's', 'p'))
        f4 = f1.assign.bloc[sel](f3 * 100)

        self.assertEqual(f4.to_pairs(), match)

    def test_frame_assign_bloc_d(self) -> None:
        records = ((1, 2, 20, 40), (30, 50, 4, 5))

        f1 = Frame.from_records(
            records,
            columns=(
                'p',
                'q',
                'r',
                's',
            ),
            index=('x', 'y'),
        )

        sel = f1 < 100  # get all true

        # get a value that will require reindexing
        f2 = Frame.from_element(
            -100,
            columns=(
                'q',
                'r',
            ),
            index=('y',),
        )

        self.assertEqual(
            f1.assign.bloc[sel](f2).to_pairs(),
            (
                ('p', (('x', 1), ('y', 30))),
                ('q', (('x', 2), ('y', -100))),
                ('r', (('x', 20), ('y', -100))),
                ('s', (('x', 40), ('y', 5))),
            ),
        )

    def test_frame_assign_bloc_e(self) -> None:
        records = ((1, 20), (30, 5))
        f1 = Frame.from_records(
            records,
            columns=(
                'p',
                'q',
            ),
            index=('x', 'y'),
        )

        with self.assertRaises(RuntimeError):
            # invalid bloc_key
            f1.assign.bloc[[[True, False], [False, True]]](3)

        with self.assertRaises(RuntimeError):
            f1.assign.bloc[np.array([[True, False, False], [False, True, True]])](3)

        with self.assertRaises(RuntimeError):
            f1.assign.bloc[np.array([[True, False], [False, True]])](
                np.array([[100, 200, 10], [200, 300, 30]])
            )

        a1 = np.array([[True, False], [False, True]])
        a2 = np.array([[100, 200], [200, 300]])
        self.assertEqual(
            f1.assign.bloc[a1](a2).to_pairs(),
            (('p', (('x', 100), ('y', 30))), ('q', (('x', 20), ('y', 300)))),
        )

    def test_frame_assign_bloc_f(self) -> None:
        f = sf.Frame.from_records(np.arange(9).reshape(3, 3))
        fgo = f.to_frame_go()
        f1 = f.assign.bloc[f % 2 == 0](-f)
        f2 = f.assign.bloc[f % 2 == 0](-fgo)

        self.assertTrue((f1.values == f2.values).all())

    def test_frame_assign_bloc_g(self) -> None:
        f = sf.Frame.from_records(
            ((None, np.datetime64('2020-01-01')), (np.datetime64('1764-01-01'), None))
        )
        f2 = f.assign.bloc[~f.isna()].apply(lambda s: s.astype('datetime64[ms]'))
        self.assertEqual(
            f2.to_pairs(),
            (
                (0, ((0, None), (1, datetime.datetime(1764, 1, 1, 0, 0)))),
                (1, ((0, datetime.datetime(2020, 1, 1, 0, 0)), (1, None))),
            ),
        )

    def test_frame_assign_bloc_h(self) -> None:
        f1 = ff.parse('s(4,8)|v(int,int,float,float,bool,bool,int,int)')
        f2 = f1.assign.bloc[f1 < 0].apply(lambda s: s * -1)
        self.assertEqual(
            [dt.kind for dt in f2.dtypes.values],
            ['f', 'f', 'f', 'f', 'b', 'b', 'f', 'f'],
        )
        self.assertEqual((f2 >= 0).values.sum(), 32)

    def test_frame_assign_bloc_i(self) -> None:
        f1 = ff.parse('s(4,8)|v(int,int,bool,bool,int,bool,int,int)')
        s1 = f1.bloc[(f1 % 2) == 1]
        self.assertEqual(
            s1.to_pairs(),
            (
                ((0, 0), -88017),
                ((0, 1), 162197),
                ((0, 2), True),
                ((0, 7), 137759),
                ((1, 0), 92867),
                ((1, 1), -41157),
                ((1, 5), True),
                ((2, 0), 84967),
                ((2, 1), 5729),
                ((2, 3), True),
                ((2, 6), 32395),
                ((3, 1), -168387),
                ((3, 2), True),
                ((3, 3), True),
                ((3, 4), 32395),
                ((3, 5), True),
                ((3, 6), 137759),
            ),
        )

        self.assertEqual(
            f1.assign.bloc[(f1 % 2) == 1]
            .apply(lambda s: s.clip(lower=-1, upper=1))
            .to_pairs(),
            (
                (0, ((0, -1), (1, 1), (2, 1), (3, 13448))),
                (1, ((0, 1), (1, -1), (2, 1), (3, -1))),
                (2, ((0, True), (1, False), (2, False), (3, True))),
                (3, ((0, False), (1, False), (2, True), (3, True))),
                (4, ((0, 58768), (1, 146284), (2, 170440), (3, 1))),
                (5, ((0, False), (1, True), (2, False), (3, True))),
                (6, ((0, 146284), (1, 170440), (2, 1), (3, 1))),
                (7, ((0, 1), (1, -62964), (2, 172142), (3, -154686))),
            ),
        )

    def test_frame_assign_bloc_j(self) -> None:
        f1 = ff.parse('s(2,4)|v(int,bool)')
        f2 = f1.assign.bloc[f1 < 0](0)
        self.assertEqual(
            f2.to_pairs(),
            (
                (0, ((0, 0), (1, 92867))),
                (1, ((0, False), (1, False))),
                (2, ((0, 0), (1, 91301))),
                (3, ((0, False), (1, False))),
            ),
        )

    def test_frame_assign_bloc_k(self) -> None:
        f1 = ff.parse('s(2,4)|v(int,int,bool)')
        f2 = ff.parse('s(2,4)|v(str)')

        f3 = f1.assign.bloc[f1 < 0](f2)

        self.assertEqual(
            f3.to_pairs(),
            (
                (0, ((0, 'zjZQ'), (1, 92867))),
                (1, ((0, 162197), (1, 'zJnC'))),
                (2, ((0, True), (1, False))),
                (3, ((0, 129017), (1, 35021))),
            ),
        )

    # ---------------------------------------------------------------------------

    def test_frame_mask_loc_a(self) -> None:
        records = ((1, 2, 'a', False, True), (30, 50, 'b', True, False))

        f1 = Frame.from_records(
            records, columns=('p', 'q', 'r', 's', 't'), index=('x', 'y')
        )

        self.assertEqual(
            f1.mask.loc['x', 'r':].values.tolist(),  # type: ignore  # https://github.com/python/typeshed/pull/3024
            [[False, False, True, True, True], [False, False, False, False, False]],
        )

        self.assertEqual(
            f1.mask['s'].values.tolist(),
            [[False, False, False, True, False], [False, False, False, True, False]],
        )

    def test_frame_masked_array_loc_a(self) -> None:
        records = ((1, 2, 'a', False, True), (30, 50, 'b', True, False))

        f1 = Frame.from_records(
            records, columns=('p', 'q', 'r', 's', 't'), index=('x', 'y')
        )

        # mask our the non-integers
        self.assertEqual(f1.masked_array.loc[:, 'r':].sum(), 83)  # type: ignore  # https://github.com/python/typeshed/pull/3024

    def test_frame_masked_array_getitem_a(self) -> None:
        records = ((1, 2, 'a', False, True), (30, 50, 'b', True, False))

        f1 = Frame.from_records(
            records, columns=('p', 'q', 'r', 's', 't'), index=('x', 'y')
        )

        self.assertEqual(
            f1.masked_array['r':].tolist(),  # type: ignore
            [[1, 2, None, None, None], [30, 50, None, None, None]],
        )

    # ---------------------------------------------------------------------------

    def test_frame_reindex_other_like_iloc_a(self) -> None:
        records = ((1, 2, 'a', False, True), (30, 50, 'b', True, False))

        f1 = Frame.from_records(
            records, columns=('p', 'q', 'r', 's', 't'), index=('x', 'y')
        )

        value1 = Series((100, 200, 300), index=('s', 'u', 't'))
        iloc_key1 = (0, slice(2, None))
        v1 = f1._reindex_other_like_iloc(
            value1,
            iloc_key1,
            is_series=True,
            is_frame=False,
        )

        self.assertAlmostEqualItems(v1.items(), [('r', nan), ('s', 100), ('t', 300)])

        value2 = Series((100, 200), index=('y', 'x'))
        iloc_key2 = (slice(0, None), 2)
        v2 = f1._reindex_other_like_iloc(
            value2,
            iloc_key2,
            is_series=True,
            is_frame=False,
        )

        self.assertAlmostEqualItems(v2.items(), [('x', 200), ('y', 100)])

    def test_frame_reindex_other_like_iloc_b(self) -> None:
        records = ((1, 2, 'a', False, True), (30, 50, 'b', True, False))

        f1 = Frame.from_records(
            records, columns=('p', 'q', 'r', 's', 't'), index=('x', 'y')
        )

        value1 = Frame.from_records(
            ((20, 21, 22), (23, 24, 25)), index=('x', 'y'), columns=('s', 't', 'w')
        )

        iloc_key1 = (slice(0, None), slice(3, None))
        v1 = f1._reindex_other_like_iloc(
            value1,
            iloc_key1,
            is_series=False,
            is_frame=True,
        )

        self.assertEqual(
            v1.to_pairs(),
            (('s', (('x', 20), ('y', 23))), ('t', (('x', 21), ('y', 24)))),
        )

    def test_frame_reindex_other_like_iloc_c(self) -> None:
        records = ((1, 2, 'a', False, True), (30, 50, 'b', True, False))

        f1 = Frame.from_records(
            records, columns=('p', 'q', 'r', 's', 't'), index=('x', 'y')
        )

        iloc_key1 = (slice(0, None), slice(3, None))

        with self.assertRaises(RuntimeError):
            _ = f1._reindex_other_like_iloc(
                Series([3, 4, 5]),
                iloc_key1,
                is_series=True,
                is_frame=False,
            )

    # ---------------------------------------------------------------------------

    def test_frame_axis_flat_a(self) -> None:
        records = (
            (1, 2, 'a', False, True),
            (30, 34, 'b', True, False),
            (54, 95, 'c', False, False),
            (65, 73, 'd', True, True),
        )

        f1 = Frame.from_records(
            records, columns=('p', 'q', 'r', 's', 't'), index=('w', 'x', 'y', 'z')
        )

        self.assertEqual(
            f1.to_pairs(axis=1),
            (
                ('w', (('p', 1), ('q', 2), ('r', 'a'), ('s', False), ('t', True))),
                ('x', (('p', 30), ('q', 34), ('r', 'b'), ('s', True), ('t', False))),
                ('y', (('p', 54), ('q', 95), ('r', 'c'), ('s', False), ('t', False))),
                ('z', (('p', 65), ('q', 73), ('r', 'd'), ('s', True), ('t', True))),
            ),
        )

        self.assertEqual(
            f1.to_pairs(axis=0),
            (
                ('p', (('w', 1), ('x', 30), ('y', 54), ('z', 65))),
                ('q', (('w', 2), ('x', 34), ('y', 95), ('z', 73))),
                ('r', (('w', 'a'), ('x', 'b'), ('y', 'c'), ('z', 'd'))),
                ('s', (('w', False), ('x', True), ('y', False), ('z', True))),
                ('t', (('w', True), ('x', False), ('y', False), ('z', True))),
            ),
        )

    # ---------------------------------------------------------------------------

    def test_frame_reindex_a(self) -> None:
        records = (
            (1, 2, 'a', False, True),
            (30, 34, 'b', True, False),
            (54, 95, 'c', False, False),
            (65, 73, 'd', True, True),
        )

        f1 = Frame.from_records(
            records, columns=('p', 'q', 'r', 's', 't'), index=('w', 'x', 'y', 'z')
        )

        # subset index reindex
        self.assertEqual(
            f1.reindex(('z', 'w')).to_pairs(axis=0),
            (
                ('p', (('z', 65), ('w', 1))),
                ('q', (('z', 73), ('w', 2))),
                ('r', (('z', 'd'), ('w', 'a'))),
                ('s', (('z', True), ('w', False))),
                ('t', (('z', True), ('w', True))),
            ),
        )

        # index only with nan filling
        self.assertEqual(
            f1.reindex(('z', 'b', 'w'), fill_value=None).to_pairs(),
            (
                ('p', (('z', 65), ('b', None), ('w', 1))),
                ('q', (('z', 73), ('b', None), ('w', 2))),
                ('r', (('z', 'd'), ('b', None), ('w', 'a'))),
                ('s', (('z', True), ('b', None), ('w', False))),
                ('t', (('z', True), ('b', None), ('w', True))),
            ),
        )

    def test_frame_reindex_b(self) -> None:
        records = (
            (1, 2, 'a', False, True),
            (30, 34, 'b', True, False),
            (54, 95, 'c', False, False),
            (65, 73, 'd', True, True),
        )

        f1 = Frame.from_records(
            records, columns=('p', 'q', 'r', 's', 't'), index=('w', 'x', 'y', 'z')
        )

        self.assertEqual(
            f1.reindex(columns=('q', 'p', 'w'), fill_value=None).to_pairs(),
            (
                ('q', (('w', 2), ('x', 34), ('y', 95), ('z', 73))),
                ('p', (('w', 1), ('x', 30), ('y', 54), ('z', 65))),
                ('w', (('w', None), ('x', None), ('y', None), ('z', None))),
            ),
        )

        self.assertEqual(
            f1.reindex(columns=('q', 'p', 's')).to_pairs(),
            (
                ('q', (('w', 2), ('x', 34), ('y', 95), ('z', 73))),
                ('p', (('w', 1), ('x', 30), ('y', 54), ('z', 65))),
                ('s', (('w', False), ('x', True), ('y', False), ('z', True))),
            ),
        )

        f2: Frame = f1[['p', 'q']]  # type: ignore

        self.assertEqual(
            f2.reindex(columns=('q', 'p')).to_pairs(),
            (
                ('q', (('w', 2), ('x', 34), ('y', 95), ('z', 73))),
                ('p', (('w', 1), ('x', 30), ('y', 54), ('z', 65))),
            ),
        )

        self.assertEqual(
            f2.reindex(columns=('a', 'b'), fill_value=None).to_pairs(),
            (
                ('a', (('w', None), ('x', None), ('y', None), ('z', None))),
                ('b', (('w', None), ('x', None), ('y', None), ('z', None))),
            ),
        )

    def test_frame_reindex_c(self) -> None:
        # reindex both axis
        records = (
            (1, 2, 'a', False, True),
            (30, 34, 'b', True, False),
            (54, 95, 'c', False, False),
            (65, 73, 'd', True, True),
        )

        f1 = Frame.from_records(
            records, columns=('p', 'q', 'r', 's', 't'), index=('w', 'x', 'y', 'z')
        )

        self.assertEqual(
            f1.reindex(index=('y', 'x'), columns=('s', 'q')).to_pairs(),
            (('s', (('y', False), ('x', True))), ('q', (('y', 95), ('x', 34)))),
        )

        self.assertEqual(
            f1.reindex(
                index=('x', 'y'), columns=('s', 'q', 'u'), fill_value=None
            ).to_pairs(),
            (
                ('s', (('x', True), ('y', False))),
                ('q', (('x', 34), ('y', 95))),
                ('u', (('x', None), ('y', None))),
            ),
        )

        self.assertEqual(
            f1.reindex(index=('a', 'b'), columns=('c', 'd'), fill_value=None).to_pairs(),
            (('c', (('a', None), ('b', None))), ('d', (('a', None), ('b', None)))),
        )

        f2: Frame = f1[['p', 'q']]  # type: ignore

        self.assertEqual(
            f2.reindex(index=('x',), columns=('q',)).to_pairs(), (('q', (('x', 34),)),)
        )

        self.assertEqual(
            f2.reindex(index=('y', 'x', 'q'), columns=('q',), fill_value=None).to_pairs(),
            (('q', (('y', 95), ('x', 34), ('q', None))),),
        )

    def test_frame_reindex_d(self) -> None:
        records = (
            (1, 2, 'a', False, True),
            (30, 34, 'b', True, False),
            (54, 95, 'c', False, False),
        )

        columns = IndexHierarchy.from_labels(
            (('a', 1), ('a', 2), ('b', 1), ('b', 2), ('b', 3))
        )
        f1 = Frame.from_records(records, columns=columns, index=('x', 'y', 'z'))

        # NOTE: must use HLoc on getting a single columns as otherwise looks like a multiple axis selection
        self.assertEqual(f1[sf.HLoc['a', 1]].to_pairs(), (('x', 1), ('y', 30), ('z', 54)))

        self.assertEqual(
            f1[sf.HLoc['b', 1] :].to_pairs(),  # type: ignore  # https://github.com/python/typeshed/pull/3024
            (
                (('b', 1), (('x', 'a'), ('y', 'b'), ('z', 'c'))),
                (('b', 2), (('x', False), ('y', True), ('z', False))),
                (('b', 3), (('x', True), ('y', False), ('z', False))),
            ),
        )

        # reindexing with no column matches results in NaN for all values
        self.assertTrue(
            f1.iloc[:, 1:]
            .reindex(columns=IndexHierarchy.from_product(('b', 'a'), (10, 20)))
            .isna()
            .all()
            .all()
        )

        columns = IndexHierarchy.from_product(('b', 'a'), (3, 2))
        f2 = f1.reindex(columns=columns, fill_value=None)
        self.assertEqual(
            f2.to_pairs(),
            (
                (('b', 3), (('x', True), ('y', False), ('z', False))),
                (('b', 2), (('x', False), ('y', True), ('z', False))),
                (('a', 3), (('x', None), ('y', None), ('z', None))),
                (('a', 2), (('x', 2), ('y', 34), ('z', 95))),
            ),
        )

    def test_frame_reindex_e(self) -> None:
        records = (
            (1, 2, 'a', False),
            (30, 34, 'b', True),
            (54, 95, 'c', False),
            (65, 73, 'd', True),
        )

        columns = IndexHierarchy.from_product(('a', 'b'), (1, 2))
        index = IndexHierarchy.from_product((100, 200), (True, False))

        f1 = Frame.from_records(records, columns=columns, index=index)

        self.assertEqual(
            f1.loc[(200, True) :, ('b', 1) :].to_pairs(),  # type: ignore  # https://github.com/python/typeshed/pull/3024
            (
                (('b', 1), (((200, True), 'c'), ((200, False), 'd'))),
                (('b', 2), (((200, True), False), ((200, False), True))),
            ),
        )

        # reindex from IndexHierarchy to Index with tuples
        f2 = f1.reindex(
            index=IndexHierarchy.from_product((200, 300), (False, True)),
            columns=[('b', 1), ('a', 1)],
            fill_value=None,
        )
        self.assertEqual(
            f2.to_pairs(),
            (
                (
                    ('b', 1),
                    (
                        ((200, False), 'd'),
                        ((200, True), 'c'),
                        ((300, False), None),
                        ((300, True), None),
                    ),
                ),
                (
                    ('a', 1),
                    (
                        ((200, False), 65),
                        ((200, True), 54),
                        ((300, False), None),
                        ((300, True), None),
                    ),
                ),
            ),
        )

    def test_frame_reindex_f(self) -> None:
        records = (
            (1, 2, 'a', False),
            (30, 34, 'b', True),
        )
        columns = IndexHierarchy.from_product(('a', 'b'), (1, 2))
        f1 = Frame.from_records(records, columns=columns, name='foo')
        f2 = f1.reindex(index=(0, 1, 2), fill_value=0)

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

        self.assertEqual(
            f2.to_pairs(),
            ((('a', 2), ((0, 2), (1, 34))), (('b', 1), ((0, 'a'), (1, 'b')))),
        )

    def test_frame_reindex_h(self) -> None:
        # reindex both axis
        records = (
            (2, 'a', False),
            (34, 'b', True),
        )
        f1 = Frame.from_records(records, columns=('p', 'q', 'r'), index=('w', 'x'))

        with self.assertRaises(RuntimeError):
            _ = f1.reindex()

    def test_frame_reindex_i(self) -> None:
        f1 = sf.Frame.from_dict(
            dict(a=[0.1, 1, 2], b=['str1', 'str2', 'str3']),
        )
        f2 = f1.reindex(columns=['a', 'c'], fill_value=np.nan)
        self.assertEqual([d.kind for d in f2.dtypes.values], ['f', 'f'])
        self.assertEqual(
            f2.fillna(-1).to_pairs(),
            (
                ('a', ((0, 0.1), (1, 1.0), (2, 2.0))),
                ('c', ((0, -1.0), (1, -1.0), (2, -1.0))),
            ),
        )

    def test_frame_reindex_j(self) -> None:
        f1 = sf.Frame.from_dict(
            dict(a=[0.1, 1, 2], b=['str1', 'str2', 'str3']),
        )
        f2 = f1.reindex(columns=('c', 'd'), fill_value=0)
        self.assertEqual([d.kind for d in f2.dtypes.values], ['i', 'i'])

        f3 = f1.reindex(columns=('c', 'd'), index=(10, 11), fill_value=0)
        self.assertEqual([d.kind for d in f3.dtypes.values], ['i', 'i'])

    def test_frame_reindex_k1(self) -> None:
        records = (
            (2, 'a', False),
            (34, 'b', True),
        )
        f1 = Frame.from_records(records, columns=('p', 'q', 'r'), index=('w', 'x'))

        f2 = f1.reindex(index=('x', 'y'), fill_value=[-1, -2, -3])
        self.assertEqual(
            f2.values.tolist(),
            [[34, 'b', True], [-1, -2, -3]],
        )

    def test_frame_reindex_k2(self) -> None:
        records = (
            (2, 'a', False),
            (34, 'b', True),
        )
        f1 = Frame.from_records(records, columns=('p', 'q', 'r'), index=('w', 'x'))

        f2 = f1.reindex(index=('x', 'y'), fill_value=FillValueAuto(i=0, U='', b=False))
        self.assertEqual(
            f2.to_pairs(),
            (
                ('p', (('x', 34), ('y', 0))),
                ('q', (('x', 'b'), ('y', ''))),
                ('r', (('x', True), ('y', False))),
            ),
        )

    def test_frame_reindex_k3(self) -> None:
        records = (
            (2, 'a', False),
            (34, 'b', True),
        )
        f1 = Frame.from_records(records, columns=('p', 'q', 'r'), index=('w', 'x'))

        f2 = f1.reindex(columns=('p', 'r', 's', 't'), fill_value=[-1, -2, -3, -4])
        self.assertEqual(
            f2.to_pairs(),
            (
                ('p', (('w', 2), ('x', 34))),
                ('r', (('w', False), ('x', True))),
                ('s', (('w', -3), ('x', -3))),
                ('t', (('w', -4), ('x', -4))),
            ),
        )

    def test_frame_reindex_k4(self) -> None:
        records = (
            (2, 'a', False),
            (34, 'b', True),
        )
        f1 = Frame.from_records(records, columns=('p', 'q', 'r'), index=('w', 'x'))

        f2 = f1.reindex(columns=('p', 'r'), index=('w',), fill_value=[-1, -2])
        self.assertEqual(f2.to_pairs(), (('p', (('w', 2),)), ('r', (('w', False),))))

    def test_frame_reindex_k5(self) -> None:
        records = (
            (2, 'a', False),
            (34, 'b', True),
        )
        f1 = Frame.from_records(records, columns=('p', 'q', 'r'), index=('w', 'x'))

        f2 = f1.reindex(
            columns=('p', 'r'), index=('w', 'y', 'x'), fill_value=FillValueAuto
        )
        self.assertEqual(
            f2.to_pairs(),
            (
                ('p', (('w', 2), ('y', 0), ('x', 34))),
                ('r', (('w', False), ('y', False), ('x', True))),
            ),
        )

    def test_frame_reindex_k6(self) -> None:
        records = (
            (2, 'a', False),
            (34, 'b', True),
        )
        f1 = Frame.from_records(records, columns=('p', 'q', 'r'), index=('w', 'x'))

        with self.assertRaises(RuntimeError):
            _ = f1.reindex(columns=('r', 's'), fill_value=FillValueAuto)

    def test_frame_reindex_k7(self) -> None:
        records = (
            (2, 'a', False),
            (34, 'b', True),
        )
        f1 = Frame.from_records(records, columns=('p', 'q', 'r'), index=('w', 'x'))

        # explicitly handle no-op scenario
        f2 = f1.reindex(index=('x',), fill_value=FillValueAuto)
        self.assertEqual(
            f2.to_pairs(),
            (('p', (('x', 34),)), ('q', (('x', 'b'),)), ('r', (('x', True),))),
        )

    def test_frame_reindex_k8(self) -> None:
        records = (
            (2, 1, 4),
            (34, 3, 5),
        )
        f1 = Frame.from_records(
            records,
            columns=('p', 'q', 'r'),
            index=('w', 'x'),
            consolidate_blocks=True,
        )

        # explicitly handle no-op scenario
        f2 = f1.reindex(columns=('p', 'r'), fill_value=FillValueAuto)
        self.assertEqual(
            f2.to_pairs(), (('p', (('w', 2), ('x', 34))), ('r', (('w', 4), ('x', 5))))
        )

    def test_frame_reindex_k9(self) -> None:
        records = (
            (2, 'x', False),
            (3, 'y', True),
        )
        f1 = Frame.from_records(
            records,
            columns=('p', 'q', 'r'),
            index=('w', 'x'),
            consolidate_blocks=True,
        )

        # explicitly handle no-op scenario
        f2 = f1.reindex(columns=('s', 't'), index=('a', 'b'), fill_value=[-1, -2])
        self.assertEqual(
            f2.to_pairs(),
            (('s', (('a', -1), ('b', -1))), ('t', (('a', -2), ('b', -2)))),
        )

    def test_frame_reindex_m(self) -> None:
        records = (
            (2, 'a', False),
            (3, 'b', True),
            (10, 'c', True),
        )
        f1 = Frame.from_records(
            records,
            columns=('p', 'q', 'r'),
            index=('w', 'x', 'y'),
            consolidate_blocks=True,
        )

        f2 = f1.reindex(columns=('r', 'p'), index=('y', 'x'))
        self.assertEqual(
            f2.to_pairs(),
            (('r', (('y', True), ('x', True))), ('p', (('y', 10), ('x', 3)))),
        )

    def test_frame_reindex_n1(self) -> None:
        records = (
            (2, '2024', '1743'),
            (3, '2022', '1832'),
            (10, '1542', '1523'),
        )
        f1 = Frame.from_records(
            records,
            columns=('p', 'q', 'r'),
            index=('w', 'x', 'y'),
            consolidate_blocks=True,
            dtypes=(int, 'datetime64[ns]', 'datetime64[ns]'),
        )

        f2 = f1.reindex(index=('y', 'x', 'z'), fill_value=None)
        self.assertEqual(
            f2.to_pairs(),
            (
                ('p', (('y', 10), ('x', 3), ('z', None))),
                (
                    'q',
                    (
                        ('y', np.datetime64('2126-07-21T23:34:33.709551616')),
                        ('x', np.datetime64('2022-01-01T00:00:00.000000000')),
                        ('z', None),
                    ),
                ),
                (
                    'r',
                    (
                        ('y', np.datetime64('2107-07-21T23:34:33.709551616')),
                        ('x', np.datetime64('1832-01-01T00:00:00.000000000')),
                        ('z', None),
                    ),
                ),
            ),
        )

    def test_frame_reindex_n2(self) -> None:
        records = (
            (2, '2024', '1743', False),
            (3, '2022', '1832', True),
            (10, '1542', '1523', False),
        )
        f1 = Frame.from_records(
            records,
            columns=('p', 'q', 'r', 's'),
            index=('w', 'x', 'y'),
            consolidate_blocks=True,
            dtypes=(int, 'datetime64[ns]', 'datetime64[ns]', bool),
        )

        f2 = f1.reindex(index=('y', 'x', 'z'), columns=('q', 'r', 'p'), fill_value=None)
        self.assertEqual(
            f2.to_pairs(),
            (
                (
                    'q',
                    (
                        ('y', np.datetime64('2126-07-21T23:34:33.709551616')),
                        ('x', np.datetime64('2022-01-01T00:00:00.000000000')),
                        ('z', None),
                    ),
                ),
                (
                    'r',
                    (
                        ('y', np.datetime64('2107-07-21T23:34:33.709551616')),
                        ('x', np.datetime64('1832-01-01T00:00:00.000000000')),
                        ('z', None),
                    ),
                ),
                ('p', (('y', 10), ('x', 3), ('z', None))),
            ),
        )

    def test_frame_reindex_n3(self) -> None:
        records = (
            (2, '2024', '1743', False),
            (3, '2022', '1832', True),
            (10, '1542', '1523', False),
        )
        f1 = Frame.from_records(
            records,
            columns=('p', 'q', 'r', 's'),
            index=('w', 'x', 'y'),
            consolidate_blocks=True,
            dtypes=(int, 'datetime64[ns]', 'datetime64[ns]', bool),
        )
        f2 = f1.reindex(
            index=('y', 'x', 'z'),
            columns=('q', 'r', 's'),
            fill_value=FillValueAuto.from_default(M=None),
        )
        self.assertEqual(
            f2['q'].values.tolist(),
            [
                np.datetime64('2126-07-21T23:34:33.709551616'),
                np.datetime64('2022-01-01T00:00:00.000000000'),
                None,
            ],
        )
        # import ipdb; ipdb.set_trace()

    # ---------------------------------------------------------------------------

    def test_frame_contains_a(self) -> None:
        f1 = Frame.from_items(
            zip(('a', 'b'), ([20, 30, 40], [80, 10, 30])), index=('x', 'y', 'z')
        )

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
        f1 = Frame.from_records(
            records, columns=('p', 'q', 'r', 's', 't'), index=('w', 'x', 'y', 'z')
        )

        post = f1.sum(axis=0)
        self.assertAlmostEqualItems(
            list(post.items()),
            [
                ('p', 64.0),
                ('q', 204.0),
                ('r', 114.0),
                ('s', 127.01999999999998),
                ('t', 172.131),
            ],
        )
        self.assertEqual(post.dtype, np.float64)

        post = f1.sum(axis=1)  # sum columns, return row index
        self.assertEqual(
            list(f1.sum(axis=1).items()),
            [
                ('w', 61.463999999999999),
                ('x', 294.72300000000001),
                ('y', 101.5),
                ('z', 223.464),
            ],
        )
        self.assertEqual(post.dtype, np.float64)

    def test_frame_sum_b(self) -> None:
        records = (
            (2, 2, 3),
            (30, 34, 60),
            (2, 95, 1),
            (30, 73, 50),
        )
        f1 = Frame.from_records(
            records, columns=('p', 'q', 'r'), index=('w', 'x', 'y', 'z')
        )

        post = f1.sum(axis=0)

        self.assertEqual(list(post.items()), [('p', 64), ('q', 204), ('r', 114)])

        self.assertEqual(
            list(f1.sum(axis=1).items()), [('w', 7), ('x', 124), ('y', 98), ('z', 153)]
        )

    def test_frame_sum_c(self) -> None:
        index = list(''.join(x) for x in it.combinations(string.ascii_lowercase, 2))

        f1 = FrameGO(index=index)
        for col in range(100):
            s = Series.from_element(col * 0.1, index=index[col : col + 20])
            f1[col] = s
        assert f1.sum().sum() == 9900.0

    def test_frame_sum_d(self) -> None:
        a1 = np.array(
            [
                (2, 2, 3, 4.23, np.nan),
                (30, 34, None, 80.6, 90.123),
            ],
            dtype=object,
        )
        f1 = Frame(
            a1,
            columns=('p', 'q', 'r', 's', 't'),
            index=('w', 'x'),
        )

        self.assertEqual(f1.sum(axis=0).values.tolist(), [32, 36, 3, 84.83, 90.123])

        self.assertEqual(f1.sum(axis=1).values.tolist(), [11.23, 234.723])

        with self.assertRaises(TypeError):
            f1.sum(skipna=False)

    def test_frame_sum_e(self) -> None:
        f1 = ff.parse('s(1,4)|v(int,float)')
        post = f1.min(axis=1, skipna=False)
        self.assertEqual(post.to_pairs(), ((0, -88017.0),))

    def test_frame_sum_f(self) -> None:
        a1 = np.array(
            [
                (2, 2, np.nan, 4.0, np.nan),
                (30, 34, None, 80.0, 90.0),
            ],
            dtype=object,
        )
        f1 = Frame(
            a1,
            columns=('p', 'q', 'r', 's', 't'),
            index=('w', 'x'),
        )
        self.assertEqual(
            f1.sum().to_pairs(),
            (('p', 32), ('q', 36), ('r', 0), ('s', 84.0), ('t', 90.0)),
        )
        self.assertEqual(
            f1.sum(allna=-1).to_pairs(),
            (('p', 32), ('q', 36), ('r', -1), ('s', 84.0), ('t', 90.0)),
        )

    # ---------------------------------------------------------------------------

    def test_frame_mean_a(self) -> None:
        a1 = np.array([[1, 2, 3, -5], [10, 50, 30, -7], [1345, 2234, 3345, -200]])
        a2 = np.array([2, 50, 2234])
        a3 = np.array([3, 30, 3345])
        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        f1 = Frame(tb1, columns=tuple('pqrstu'), index=('w', 'x', 'y'))

        self.assertEqual(f1.mean(axis=0).values.tolist(), f1.values.mean(axis=0).tolist())

        self.assertEqual(f1.mean(axis=1).values.tolist(), f1.values.mean(axis=1).tolist())

    # ---------------------------------------------------------------------------

    def test_frame_median_a(self) -> None:
        a1 = np.array([[1, 2, 3, -5], [10, 50, 30, -7], [1345, 2234, 3345, -200]])
        a2 = np.array([2, 50, 2234])
        a3 = np.array([3, 30, 3345])
        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        f1 = Frame(tb1, columns=tuple('pqrstu'), index=('w', 'x', 'y'))

        self.assertEqual(
            f1.median(axis=0).values.tolist(), np.median(f1.values, axis=0).tolist()
        )

        self.assertEqual(
            f1.median(axis=1).values.tolist(), np.median(f1.values, axis=1).tolist()
        )

    def test_frame_std_a(self) -> None:
        a1 = np.array([[1, 2, 3, -5], [10, 50, 30, -7], [1345, 2234, 3345, -200]])
        a2 = np.array([2, 50, 2234])
        a3 = np.array([3, 30, 3345])
        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        f1 = Frame(tb1, columns=tuple('pqrstu'), index=('w', 'x', 'y'))

        self.assertEqual(
            f1.std(axis=0).values.tolist(), np.std(f1.values, axis=0).tolist()
        )

        self.assertEqual(
            f1.std(axis=1).values.tolist(), np.std(f1.values, axis=1).tolist()
        )

    def test_frame_std_b(self) -> None:
        f1 = Frame(np.arange(1, 21).reshape(4, 5))
        self.assertEqual(
            round(f1.std(), 2).values.tolist(), [5.59, 5.59, 5.59, 5.59, 5.59]
        )
        self.assertEqual(
            round(f1.std(ddof=1), 2).values.tolist(), [6.45, 6.45, 6.45, 6.45, 6.45]
        )

    # ---------------------------------------------------------------------------

    def test_frame_var_a(self) -> None:
        a1 = np.array([[1, 2, 3, -5], [10, 50, 30, -7], [1345, 2234, 3345, -200]])
        a2 = np.array([2, 50, 2234])
        a3 = np.array([3, 30, 3345])
        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        f1 = Frame(tb1, columns=tuple('pqrstu'), index=('w', 'x', 'y'))

        self.assertEqual(
            f1.var(axis=0).values.tolist(), np.var(f1.values, axis=0).tolist()
        )

        self.assertEqual(
            f1.var(axis=1).values.tolist(), np.var(f1.values, axis=1).tolist()
        )

    def test_frame_var_b(self) -> None:
        f1 = Frame(np.arange(1, 21).reshape(4, 5))

        self.assertEqual(
            round(f1.var(), 2).values.tolist(), [31.25, 31.25, 31.25, 31.25, 31.25]
        )
        self.assertEqual(
            round(f1.var(ddof=1), 2).values.tolist(),
            [41.67, 41.67, 41.67, 41.67, 41.67],
        )

    # ---------------------------------------------------------------------------

    def test_frame_prod_a(self) -> None:
        records = (
            (2, 2, 3),
            (30, 34, 60),
            (2, 95, 1),
            (30, 73, 50),
        )
        f1 = Frame.from_records(
            records, columns=('p', 'q', 'r'), index=('w', 'x', 'y', 'z')
        )

        self.assertEqual(
            f1.prod(axis=0).to_pairs(), (('p', 3600), ('q', 471580), ('r', 9000))
        )

        self.assertEqual(
            f1.prod(axis=1).to_pairs(),
            (('w', 12), ('x', 61200), ('y', 190), ('z', 109500)),
        )

    def test_frame_prod_b(self) -> None:
        a1 = np.array([[1, 2, 3, -5], [10, 50, 30, -7], [1345, 2234, 3345, -200]])
        a2 = np.array([2, 50, 2234])
        a3 = np.array([3, 30, 3345.2])
        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        f1 = Frame(tb1, columns=tuple('pqrstu'), index=('w', 'x', 'y'))

        self.assertEqual(
            f1.prod(axis=0).values.tolist(), np.prod(f1.values, axis=0).tolist()
        )

        self.assertEqual(
            f1.prod(axis=1).values.tolist(), np.prod(f1.values, axis=1).tolist()
        )

    def test_frame_prod_c(self) -> None:
        f1 = Frame.from_records([[1, np.nan, 3], [1, np.nan, 2]])
        self.assertEqual(f1.prod().values.tolist(), [1, 1, 6])
        self.assertEqual(f1.prod(allna=-1).values.tolist(), [1, -1, 6])

    # ---------------------------------------------------------------------------

    def test_frame_cumsum_a(self) -> None:
        records = (
            (2, 2, 3),
            (30, 34, 60),
            (2, 95, 1),
            (30, 73, 50),
        )
        f1 = Frame.from_records(
            records, columns=('p', 'q', 'r'), index=('w', 'x', 'y', 'z')
        )

        f2 = f1.cumsum()

        self.assertEqual(
            f2.to_pairs(),
            (
                ('p', (('w', 2), ('x', 32), ('y', 34), ('z', 64))),
                ('q', (('w', 2), ('x', 36), ('y', 131), ('z', 204))),
                ('r', (('w', 3), ('x', 63), ('y', 64), ('z', 114))),
            ),
        )
        self.assertEqual(
            f1.cumsum(axis=1).to_pairs(),
            (
                ('p', (('w', 2), ('x', 30), ('y', 2), ('z', 30))),
                ('q', (('w', 4), ('x', 64), ('y', 97), ('z', 103))),
                ('r', (('w', 7), ('x', 124), ('y', 98), ('z', 153))),
            ),
        )

    def test_frame_cumsum_b(self) -> None:
        records = (
            (2, 2, 3),
            (30, 34, 60),
            (2, np.nan, 1),
            (30, np.nan, 50),
        )
        f1 = Frame.from_records(
            records, columns=('p', 'q', 'r'), index=('w', 'x', 'y', 'z')
        )

        f2 = f1.cumsum(skipna=False)

        self.assertEqual(
            f1.cumsum(skipna=False, axis=1).fillna(None).to_pairs(),
            (
                ('p', (('w', 2.0), ('x', 30.0), ('y', 2.0), ('z', 30.0))),
                ('q', (('w', 4.0), ('x', 64.0), ('y', None), ('z', None))),
                ('r', (('w', 7.0), ('x', 124.0), ('y', None), ('z', None))),
            ),
        )

    def test_frame_cumprod_a(self) -> None:
        records = (
            (2, 2, 3),
            (30, 34, 60),
            (2, 95, 1),
            (30, 73, 50),
        )
        f1 = Frame.from_records(
            records, columns=('p', 'q', 'r'), index=('w', 'x', 'y', 'z')
        )

        self.assertEqual(
            f1.cumprod(axis=0).to_pairs(),
            (
                ('p', (('w', 2), ('x', 60), ('y', 120), ('z', 3600))),
                ('q', (('w', 2), ('x', 68), ('y', 6460), ('z', 471580))),
                ('r', (('w', 3), ('x', 180), ('y', 180), ('z', 9000))),
            ),
        )

        self.assertEqual(
            f1.cumprod(axis=1).to_pairs(),
            (
                ('p', (('w', 2), ('x', 30), ('y', 2), ('z', 30))),
                ('q', (('w', 4), ('x', 1020), ('y', 190), ('z', 2190))),
                ('r', (('w', 12), ('x', 61200), ('y', 190), ('z', 109500))),
            ),
        )

    def test_frame_min_a(self) -> None:
        # reindex both axis
        records = (
            (2, 2, 3, 4.23, 50.234),
            (30, 34, 60, 80.6, 90.123),
            (2, 95, 1, 1.96, 1.54),
            (30, 73, 50, 40.23, 30.234),
        )
        f1 = Frame.from_records(
            records, columns=('p', 'q', 'r', 's', 't'), index=('w', 'x', 'y', 'z')
        )

        self.assertAlmostEqualItems(
            tuple(f1.min().items()),
            (('p', 2.0), ('q', 2.0), ('r', 1.0), ('s', 1.96), ('t', 1.54)),
        )

        self.assertAlmostEqualItems(
            tuple(f1.min(axis=1).items()),
            (('w', 2.0), ('x', 30.0), ('y', 1.0), ('z', 30.0)),
        )

    def test_frame_max_a(self) -> None:
        # Address bug reported in #1072
        data = np.arange(9).reshape(3, 3).astype('datetime64[D]')
        data[1, 1] = np.datetime64('NAT')
        frame = sf.Frame.from_records(data)

        max_date = np.datetime64('1970-01-09', 'D')

        assert frame.max().max() == max_date
        assert frame.max().max(skipna=False) == max_date
        assert frame.max(skipna=False).max() == max_date
        assert np.isnat(frame.max(skipna=False).max(skipna=False))

    @skip_win
    def test_frame_row_dtype_a(self) -> None:
        # reindex both axis
        records = (
            (2, 2, 3, 4.23, 50.234),
            (30, 34, 60, 80.6, 90.123),
            (2, 95, 1, 1.96, 1.54),
            (30, 73, 50, 40.23, 30.234),
        )
        f1 = Frame.from_records(
            records, columns=('p', 'q', 'r', 's', 't'), index=('w', 'x', 'y', 'z')
        )

        self.assertEqual(f1['t'].dtype, np.float64)  # type: ignore
        self.assertEqual(f1['p'].dtype, np.int64)  # type: ignore

        self.assertEqual(f1.loc['w'].dtype, np.float64)  # type: ignore
        self.assertEqual(f1.loc['z'].dtype, np.float64)  # type: ignore

        self.assertEqual(f1[['r', 's']].values.dtype, np.float64)

    def test_frame_unary_operator_a(self) -> None:
        records = (
            (2, 2, 3, False, True),
            (30, 34, 60, True, False),
            (2, 95, 1, True, True),
            (30, 73, 50, False, False),
        )
        f1 = Frame.from_records(
            records, columns=('p', 'q', 'r', 's', 't'), index=('w', 'x', 'y', 'z')
        )

        # raises exception with NP14
        # self.assertEqual((-f1).to_pairs(),
        #         (('p', (('w', -2), ('x', -30), ('y', -2), ('z', -30))), ('q', (('w', -2), ('x', -34), ('y', -95), ('z', -73))), ('r', (('w', -3), ('x', -60), ('y', -1), ('z', -50))), ('s', (('w', True), ('x', False), ('y', False), ('z', True))), ('t', (('w', False), ('x', True), ('y', False), ('z', True)))))

        self.assertEqual(
            (~f1).to_pairs(),
            (
                ('p', (('w', -3), ('x', -31), ('y', -3), ('z', -31))),
                ('q', (('w', -3), ('x', -35), ('y', -96), ('z', -74))),
                ('r', (('w', -4), ('x', -61), ('y', -2), ('z', -51))),
                ('s', (('w', True), ('x', False), ('y', False), ('z', True))),
                ('t', (('w', False), ('x', True), ('y', False), ('z', True))),
            ),
        )

    # ---------------------------------------------------------------------------
    def test_frame_abs_a(self) -> None:
        records = (
            (-2, -2, -3.5),
            (-30, -34, -60.2),
            (-2, 95, -1.2),
            (-30, -73, -50.2),
        )
        f1 = Frame.from_records(
            records, columns=('p', 'q', 'r'), index=('w', 'x', 'y', 'z')
        )

        self.assertEqual(
            (f1.abs()).to_pairs(),
            (
                ('p', (('w', 2), ('x', 30), ('y', 2), ('z', 30))),
                ('q', (('w', 2), ('x', 34), ('y', 95), ('z', 73))),
                ('r', (('w', 3.5), ('x', 60.2), ('y', 1.2), ('z', 50.2))),
            ),
        )

    # ---------------------------------------------------------------------------

    def test_frame_binary_operator_a(self) -> None:
        # constants

        records = (
            (2, 2, 3.5),
            (30, 34, 60.2),
            (2, 95, 1.2),
            (30, 73, 50.2),
        )
        f1 = Frame.from_records(
            records, columns=('p', 'q', 'r'), index=('w', 'x', 'y', 'z')
        )

        self.assertEqual(
            (f1 * 30).to_pairs(),
            (
                ('p', (('w', 60), ('x', 900), ('y', 60), ('z', 900))),
                ('q', (('w', 60), ('x', 1020), ('y', 2850), ('z', 2190))),
                ('r', (('w', 105.0), ('x', 1806.0), ('y', 36.0), ('z', 1506.0))),
            ),
        )

    def test_frame_binary_operator_b(self) -> None:
        records = (
            (2, 2, 3.5),
            (30, 34, 60.2),
            (2, 95, 1.2),
            (30, 73, 50.2),
        )
        f1 = Frame.from_records(
            records, columns=('p', 'q', 'r'), index=('w', 'x', 'y', 'z')
        )

        f2 = f1.loc[['y', 'z'], ['r']]
        f3 = f1 * f2

        self.assertAlmostEqualItems(
            list(f3['p'].items()), [('w', nan), ('x', nan), ('y', nan), ('z', nan)]
        )
        self.assertAlmostEqualItems(
            list(f3['r'].items()),
            [
                ('w', nan),
                ('x', nan),
                ('y', 1.4399999999999999),
                ('z', 2520.0400000000004),
            ],
        )

    def test_frame_binary_operator_c(self) -> None:
        records = (
            (2, 2, 3.5),
            (30, 34, 60.2),
            (2, 95, 1.2),
            (30, 73, 50.2),
        )
        f1 = Frame.from_records(
            records, columns=('p', 'q', 'r'), index=('w', 'x', 'y', 'z')
        )

        s1 = Series([0, 1, 2], index=('r', 'q', 'p'))

        self.assertEqual(
            (f1 * s1).to_pairs(),
            (
                ('p', (('w', 4), ('x', 60), ('y', 4), ('z', 60))),
                ('q', (('w', 2), ('x', 34), ('y', 95), ('z', 73))),
                ('r', (('w', 0.0), ('x', 0.0), ('y', 0.0), ('z', 0.0))),
            ),
        )

        self.assertEqual(
            (f1 * [0, 1, 0]).to_pairs(),
            (
                ('p', (('w', 0), ('x', 0), ('y', 0), ('z', 0))),
                ('q', (('w', 2), ('x', 34), ('y', 95), ('z', 73))),
                ('r', (('w', 0.0), ('x', 0.0), ('y', 0.0), ('z', 0.0))),
            ),
        )

    def test_frame_binary_operator_d(self) -> None:
        records = ((2, True, ''),)
        f1 = Frame.from_records(records, columns=('p', 'q', 'r'))

        self.assertEqual((f1['q'] == True).to_pairs(), ((0, True),))

        self.assertEqual((f1['r'] == True).to_pairs(), ((0, False),))

    def test_frame_binary_operator_e(self) -> None:
        # keep column order when columns are the same
        f = sf.Frame.from_element(1, columns=['dog', 3, 'bat'], index=[1, 2])
        post = f / f.sum()
        self.assertEqual(post.columns.values.tolist(), f.columns.values.tolist())

    def test_frame_binary_operator_f(self) -> None:
        # matrix multiplication, with realigment of same sized axis

        a = Frame.from_dict(dict(a=(1, 2, 3, 4), b=(5, 6, 7, 8)), index=tuple('wxyz'))
        b = Frame.from_dict(dict(p=(1, 2), q=(3, 4), r=(5, 6)), index=tuple('ab'))

        post1 = a @ b

        self.assertEqual(
            post1.to_pairs(),
            (
                ('p', (('w', 11), ('x', 14), ('y', 17), ('z', 20))),
                ('q', (('w', 23), ('x', 30), ('y', 37), ('z', 44))),
                ('r', (('w', 35), ('x', 46), ('y', 57), ('z', 68))),
            ),
        )

        # with reorded index on b, we get the same result, as opposite axis align
        post2 = a @ b.reindex(index=('b', 'a'))

        self.assertEqual(
            post2.to_pairs(),
            (
                ('p', (('w', 11), ('x', 14), ('y', 17), ('z', 20))),
                ('q', (('w', 23), ('x', 30), ('y', 37), ('z', 44))),
                ('r', (('w', 35), ('x', 46), ('y', 57), ('z', 68))),
            ),
        )

        post3 = a @ Series([1, 2], index=tuple('ab'))
        self.assertEqual(post3.to_pairs(), (('w', 11), ('x', 14), ('y', 17), ('z', 20)))

        # index is aligned
        post4 = a @ Series([2, 1], index=tuple('ba'))
        self.assertEqual(post3.to_pairs(), (('w', 11), ('x', 14), ('y', 17), ('z', 20)))

        with self.assertRaises(NotImplementedError):
            # cannot do an operation on a 1D array
            _ = 3 @ b

        post5 = [3, 4] @ b
        self.assertEqual(post5.to_pairs(), (('p', 11), ('q', 25), ('r', 39)))

    def test_frame_binary_operator_g(self) -> None:
        # matrix multiplication, with realigment of different sized axis

        a = FrameGO.from_dict(dict(a=(1, 2, 3, 4), b=(5, 6, 7, 8)), index=tuple('wxyz'))
        a['c'] = 30

        b = Frame.from_dict(dict(p=(1, 2), q=(3, 4), r=(5, 6)), index=tuple('ab'))

        with self.assertRaises(RuntimeError):
            post1 = a @ b
            # all values would go to NaN

        a = FrameGO.from_dict(dict(a=(1, 2, 3, 4), b=(5, 6, 7, 8)), index=tuple('wxyz'))

        b = Frame.from_dict(
            dict(p=(1, 2, 3), q=(3, 4, 5), r=(5, 6, 7)), index=tuple('abc')
        )

        with self.assertRaises(RuntimeError):
            post2 = a @ b

    def test_frame_binary_operator_h(self) -> None:
        a = Frame.from_dict(dict(a=(1, 2, 3, 4), b=(5, 6, 7, 8)), index=tuple('wxyz'))
        b = Frame.from_dict(dict(p=(1, 2), q=(3, 4), r=(5, 6)), index=tuple('ab'))

        # NOTE: the following yields a ValueError from the interpreter
        # post2 = a.values @ b

        self.assertEqual(
            (a @ b.values).to_pairs(),
            (
                (0, (('w', 11), ('x', 14), ('y', 17), ('z', 20))),
                (1, (('w', 23), ('x', 30), ('y', 37), ('z', 44))),
                (2, (('w', 35), ('x', 46), ('y', 57), ('z', 68))),
            ),
        )

    def test_frame_binary_operator_i(self) -> None:
        a = sf.Frame.from_elements((1, 2, 3))
        post = a == a.to_frame_go()

        self.assertEqual(post.__class__, FrameGO)
        self.assertEqual(post.to_pairs(), ((0, ((0, True), (1, True), (2, True))),))

    def test_frame_binary_operator_j(self) -> None:
        f1 = sf.FrameGO.from_element('q', index=range(3), columns=('x', 'y'))

        f1['z'] = 'foo'

        f2 = f1 + '_'
        self.assertEqual(
            f2.to_pairs(),
            (
                ('x', ((0, 'q_'), (1, 'q_'), (2, 'q_'))),
                ('y', ((0, 'q_'), (1, 'q_'), (2, 'q_'))),
                ('z', ((0, 'foo_'), (1, 'foo_'), (2, 'foo_'))),
            ),
        )
        self.assertEqual(
            f2.dtypes.values.tolist(),
            [np.dtype('<U2'), np.dtype('<U2'), np.dtype('<U4')],
        )

        f3 = '_' + f1
        self.assertEqual(
            f3.to_pairs(),
            (
                ('x', ((0, '_q'), (1, '_q'), (2, '_q'))),
                ('y', ((0, '_q'), (1, '_q'), (2, '_q'))),
                ('z', ((0, '_foo'), (1, '_foo'), (2, '_foo'))),
            ),
        )

        f4 = f1 * 3
        self.assertEqual(
            f4.to_pairs(),
            (
                ('x', ((0, 'qqq'), (1, 'qqq'), (2, 'qqq'))),
                ('y', ((0, 'qqq'), (1, 'qqq'), (2, 'qqq'))),
                ('z', ((0, 'foofoofoo'), (1, 'foofoofoo'), (2, 'foofoofoo'))),
            ),
        )
        self.assertEqual(
            f4.dtypes.values.tolist(),
            [np.dtype('<U3'), np.dtype('<U3'), np.dtype('<U9')],
        )

    def test_frame_binary_operator_k(self) -> None:
        # handling of name attr

        f1 = Frame.from_dict(
            dict(a=(1, 2, 3), b=(5, 6, 7)), index=tuple('xyz'), name='foo'
        )

        f2 = f1 * [[3, 5], [0, 0], [1, 1]]

        self.assertEqual(
            f2.to_pairs(),
            (
                ('a', (('x', 3), ('y', 0), ('z', 3))),
                ('b', (('x', 25), ('y', 0), ('z', 7))),
            ),
        )
        self.assertEqual(f2.name, None)

        f3 = f1 * 20
        self.assertEqual(f3.name, 'foo')

        f4 = f1 * np.array([[3, 5], [0, 0], [1, 1]])

        self.assertEqual(
            f4.to_pairs(),
            (
                ('a', (('x', 3), ('y', 0), ('z', 3))),
                ('b', (('x', 25), ('y', 0), ('z', 7))),
            ),
        )
        self.assertEqual(f2.name, None)

    def test_frame_binary_operator_l(self) -> None:
        f1 = Frame.from_dict(
            dict(a=(1, '2', 3), b=(5, '6', 7)), index=tuple('xyz'), name='foo'
        )

        # test comparison to a single string
        self.assertEqual(
            (f1 == '2').to_pairs(),
            (
                ('a', (('x', False), ('y', True), ('z', False))),
                ('b', (('x', False), ('y', False), ('z', False))),
            ),
        )

        # should be all true of we do our array conversion right
        f2 = f1 == f1.values.tolist()
        self.assertTrue(f2.all().all())

    def test_frame_binary_operator_m(self) -> None:
        # reindex both axis
        records = (
            (2, 2, 'a', False, False),
            (30, 34, 'b', True, False),
            (2, 95, 'c', False, False),
            (30, 73, 'd', True, True),
        )

        f1 = Frame.from_records(
            records,
            columns=('p', 'q', 'r', 's', 't'),
            index=('w', 'x', 'y', 'z'),
        )

        f2 = f1 == True
        self.assertEqual(
            f2.to_pairs(),
            (
                ('p', (('w', False), ('x', False), ('y', False), ('z', False))),
                ('q', (('w', False), ('x', False), ('y', False), ('z', False))),
                ('r', (('w', False), ('x', False), ('y', False), ('z', False))),
                ('s', (('w', False), ('x', True), ('y', False), ('z', True))),
                ('t', (('w', False), ('x', False), ('y', False), ('z', True))),
            ),
        )

        f3 = f1 == 2
        self.assertEqual(
            f3.to_pairs(),
            (
                ('p', (('w', True), ('x', False), ('y', True), ('z', False))),
                ('q', (('w', True), ('x', False), ('y', False), ('z', False))),
                ('r', (('w', False), ('x', False), ('y', False), ('z', False))),
                ('s', (('w', False), ('x', False), ('y', False), ('z', False))),
                ('t', (('w', False), ('x', False), ('y', False), ('z', False))),
            ),
        )

        f4 = f1 == None
        self.assertFalse(f4.any().any())

    def test_frame_binary_operator_n(self) -> None:
        # NOTE: testing FrameGO
        f1 = ff.parse('s(2,3)|v(int64)|f(Fg)')
        f2 = f1 - f1.mean(axis=0)
        self.assertEqual(
            f2.to_pairs(),
            (
                (0, ((0, -90442.0), (1, 90442.0))),
                (1, ((0, 101677.0), (1, -101677.0))),
                (2, ((0, -47474.5), (1, 47474.5))),
            ),
        )

    def test_frame_binary_operator_o(self) -> None:
        # reindex both axis
        records = ((1, 2), (5, 10))
        f1 = Frame.from_records(
            records,
            columns=('p', 'q'),
            index=('w', 'x'),
        )
        self.assertTrue((np.int64(10) * f1).equals(10 * f1))
        self.assertTrue((np.int64(10) + f1).equals(10 + f1))
        self.assertTrue((np.int64(10) / f1).equals(10 / f1))
        self.assertTrue((np.int64(10) // f1).equals(10 // f1))
        self.assertTrue((np.int64(10) - f1).equals(10 - f1))
        self.assertTrue((np.int64(5) > f1).equals(5 > f1))  # type: ignore
        self.assertTrue((np.int64(5) >= f1).equals(5 >= f1))  # type: ignore
        self.assertTrue((np.int64(5) <= f1).equals(5 <= f1))  # type: ignore
        self.assertTrue((np.int64(5) < f1).equals(5 < f1))  # type: ignore
        self.assertTrue((np.int64(5) == f1).equals(5 == f1))
        self.assertTrue((np.int64(5) != f1).equals(5 != f1))

    # ---------------------------------------------------------------------------

    def test_frame_isin_a(self) -> None:
        # reindex both axis
        records = (
            (2, 2, 'a', False, False),
            (30, 34, 'b', True, False),
            (2, 95, 'c', False, False),
            (30, 73, 'd', True, True),
        )

        f1 = Frame.from_records(
            records, columns=('p', 'q', 'r', 's', 't'), index=('w', 'x', 'y', 'z')
        )

        post = f1.isin({'a', 73, 30})
        self.assertEqual(
            post.to_pairs(),
            (
                ('p', (('w', False), ('x', True), ('y', False), ('z', True))),
                ('q', (('w', False), ('x', False), ('y', False), ('z', True))),
                ('r', (('w', True), ('x', False), ('y', False), ('z', False))),
                ('s', (('w', False), ('x', False), ('y', False), ('z', False))),
                ('t', (('w', False), ('x', False), ('y', False), ('z', False))),
            ),
        )

        post = f1.isin(['a', 73, 30])
        self.assertEqual(
            post.to_pairs(),
            (
                ('p', (('w', False), ('x', True), ('y', False), ('z', True))),
                ('q', (('w', False), ('x', False), ('y', False), ('z', True))),
                ('r', (('w', True), ('x', False), ('y', False), ('z', False))),
                ('s', (('w', False), ('x', False), ('y', False), ('z', False))),
                ('t', (('w', False), ('x', False), ('y', False), ('z', False))),
            ),
        )

    def test_frame_isin_b(self) -> None:
        f1 = Frame.from_fields(
            (
                ['a', 'b'],
                [True, False],
                [np.datetime64('2012'), np.datetime64('2020')],
            )
        )

        post1 = f1.isin((np.datetime64('2020'),))
        self.assertEqual(
            post1.to_pairs(),
            (
                (0, ((0, False), (1, False))),
                (1, ((0, False), (1, False))),
                (2, ((0, False), (1, True))),
            ),
        )

        post2 = f1.isin(())
        self.assertEqual(
            post2.to_pairs(),
            (
                (0, ((0, False), (1, False))),
                (1, ((0, False), (1, False))),
                (2, ((0, False), (1, False))),
            ),
        )

    def test_frame_isin_c(self) -> None:
        f1 = Frame.from_element(9223372036854775808, index=(1, 2), columns=(3, 4))
        f2 = f1.isin((9223372036854775808,))
        self.assertEqual(
            f2.to_pairs(), ((3, ((1, True), (2, True))), (4, ((1, True), (2, True))))
        )

    def test_frame_isin_d(self) -> None:
        # NOTE: the usage of True/Falseint8 in the second column was made this fail
        f1 = Frame.from_fields(
            [(9223372036854775808, 0), (True, False)],
            index=(1, 2),
            columns=(3, 4),
            dtypes=(np.uint64, np.int8),
        )

        f2 = f1.isin((9223372036854775808,))
        self.assertEqual(
            f2.to_pairs(),
            ((3, ((1, True), (2, False))), (4, ((1, False), (2, False)))),
        )

    # ---------------------------------------------------------------------------

    def test_frame_transpose_a(self) -> None:
        # reindex both axis
        records = (
            (2, 2, 'a', False, False),
            (30, 34, 'b', True, False),
            (2, 95, 'c', False, False),
            (30, 73, 'd', True, True),
        )

        f1 = Frame.from_records(
            records,
            columns=('p', 'q', 'r', 's', 't'),
            index=('w', 'x', 'y', 'z'),
            name='foo',
        )

        f2 = f1.transpose()

        self.assertEqual(
            f2.to_pairs(),
            (
                ('w', (('p', 2), ('q', 2), ('r', 'a'), ('s', False), ('t', False))),
                ('x', (('p', 30), ('q', 34), ('r', 'b'), ('s', True), ('t', False))),
                ('y', (('p', 2), ('q', 95), ('r', 'c'), ('s', False), ('t', False))),
                ('z', (('p', 30), ('q', 73), ('r', 'd'), ('s', True), ('t', True))),
            ),
        )

        self.assertEqual(f2.name, f1.name)

    def test_frame_transpose_b(self) -> None:
        # reindex both axis
        records = (
            (False, False),
            (True, False),
        )

        f1 = FrameGO.from_records(
            records, columns=IndexYearGO(('2019', '2020')), index=('x', 'y'), name='foo'
        )

        f1['2021'] = True
        self.assertTrue(
            f1.to_pairs(),
            (
                (np.datetime64('2019'), (('x', False), ('y', True))),
                (np.datetime64('2020'), (('x', False), ('y', False))),
                (np.datetime64('2021'), (('x', True), ('y', True))),
            ),
        )
        self.assertTrue(
            f1.T.to_pairs(),
            (
                (
                    'x',
                    (
                        (np.datetime64('2019'), False),
                        (np.datetime64('2020'), False),
                        (np.datetime64('2021'), True),
                    ),
                ),
                (
                    'y',
                    (
                        (np.datetime64('2019'), True),
                        (np.datetime64('2020'), False),
                        (np.datetime64('2021'), True),
                    ),
                ),
            ),
        )

    def test_frame_transpose_c(self) -> None:
        a1 = np.arange(30)
        a2 = a1.reshape(5, 6)
        a2.flags.writeable = False
        f1 = Frame(a2)
        self.assertEqual(id(a2), id(f1.values))
        self.assertEqual(id(a1), id(f1.values.base))

        f2 = f1.T
        self.assertEqual(id(a1), id(f2.values.base))

    # ---------------------------------------------------------------------------

    def test_frame_from_element_items_a(self) -> None:
        items = (((0, 1), 'g'), ((1, 0), 'q'))

        f1 = Frame.from_element_items(
            items,
            index=range(2),
            columns=range(2),
            dtype=object,
            name='foo',
            index_constructor=IndexDefaultConstructorFactory('bar'),
            columns_constructor=IndexDefaultConstructorFactory('baz'),
        )

        self.assertEqual(
            f1.to_pairs(), ((0, ((0, None), (1, 'q'))), (1, ((0, 'g'), (1, None))))
        )

        self.assertEqual(f1.name, 'foo')
        self.assertEqual(f1.index.name, 'bar')
        self.assertEqual(f1.columns.name, 'baz')

    def test_frame_from_element_items_b(self) -> None:
        items = (((0, 1), 0.5), ((1, 0), 1.5))

        f2 = Frame.from_element_items(
            items, index=range(2), columns=range(2), dtype=float
        )

        self.assertAlmostEqualItems(tuple(f2[0].items()), ((0, nan), (1, 1.5)))

        self.assertAlmostEqualItems(tuple(f2[1].items()), ((0, 0.5), (1, nan)))

    def test_frame_from_element_items_c(self) -> None:
        items = (((0, 1), 'g'), ((1, 0), 'q'))

        with self.assertRaises(AxisInvalid):
            f1 = Frame.from_element_items(
                items,
                index=range(2),
                columns=range(2),
                dtype=str,
                axis=3,
            )

    def test_frame_from_element_items_d(self) -> None:
        items = (
            ((0, 1), 'g'),
            ((1, 0), 'q'),
            ((1, 1), 'a'),
            ((0, 0), 'b'),
        )
        with self.assertRaises(ErrorInitFrame):
            _ = Frame.from_element_items(
                items,
                index=range(2),
                columns=range(2),
                axis=None,
                dtype=(int, float),
            )

    def test_frame_from_element_items_e(self) -> None:
        items = (((0, 1), True), ((1, 0), True))

        with self.assertRaises(InvalidFillValue):
            f2 = Frame.from_element_items(
                items,
                index=range(2),
                columns=range(2),
                dtype=bool,
                fill_value=FillValueAuto,
            )

    # ---------------------------------------------------------------------------

    def test_frame_from_element_loc_items_a(self) -> None:
        items = ((('b', 'x'), 'g'), (('a', 'y'), 'q'))

        f1 = Frame.from_element_items(
            items, index=('a', 'b'), columns=('x', 'y'), dtype=object, name='foo'
        )

        self.assertEqual(
            f1.to_pairs(),
            (('x', (('a', None), ('b', 'g'))), ('y', (('a', 'q'), ('b', None)))),
        )
        self.assertEqual(f1.name, 'foo')

    def test_frame_from_element_loc_items_b(self) -> None:
        records = (
            (
                2,
                2,
                'a',
                False,
            ),
            (
                30,
                34,
                'b',
                True,
            ),
            (
                2,
                95,
                'c',
                False,
            ),
        )

        f1 = Frame.from_records(
            records,
            columns=(
                'p',
                'q',
                'r',
                's',
            ),
            index=('w', 'x', 'y'),
        )

        items = f1.iter_element_items(axis=0)

        f2 = Frame.from_element_items(
            items,
            index=f1.index,
            columns=f1.columns,
            axis=0,
        )

    # ---------------------------------------------------------------------------

    def test_frame_from_items_a(self) -> None:
        f1 = Frame.from_items(
            zip(range(10), (np.random.rand(1000) for _ in range(10))), name='foo'
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

        self.assertEqual(
            f1.to_pairs(),
            (
                ('x', (('a', 1), ('b', 2), ('c', 3))),
                ('y', (('a', 4), ('b', 5), ('c', 6))),
            ),
        )

    def test_frame_from_items_d(self) -> None:
        s1 = Series((1, 2, 3), index=('a', 'b', 'c'))
        s2 = Series((4, 5, 6), index=('a', 'b', 'c'))

        f1 = Frame.from_items(zip(list('xy'), (s1, s2)), index=('c', 'a'))

        self.assertEqual(
            f1.to_pairs(), (('x', (('c', 3), ('a', 1))), ('y', (('c', 6), ('a', 4))))
        )

    def test_frame_from_items_e(self) -> None:
        s1 = Series((1, 2, 3), index=('a', 'b', 'c'))
        s2 = Series((4, 5, 6), index=('a', 'b', 'c'))
        s3 = Series((7, 8, 9), index=('a', 'b', 'c'))

        f1 = Frame.from_items(
            zip(list('xy'), (s1, s2, s3)), index=('c', 'a'), consolidate_blocks=True
        )

        self.assertEqual(len(f1._blocks._blocks), 1)

    def test_frame_from_items_f(self) -> None:
        def gen() -> tp.Iterator[tp.Tuple[int, tp.Tuple[int, int]]]:
            for i in range(4):
                yield i, (2 * i, 3 * i)

        f1 = Frame.from_items(gen(), name='foo', dtypes=(str, str, str, str))
        self.assertEqual(
            f1.to_pairs(),
            (
                (0, ((0, '0'), (1, '0'))),
                (1, ((0, '2'), (1, '3'))),
                (2, ((0, '4'), (1, '6'))),
                (3, ((0, '6'), (1, '9'))),
            ),
        )

    def test_frame_from_items_g(self) -> None:
        def gen() -> tp.Iterator[tp.Tuple[tp.Tuple[str, int], tp.Tuple[int, int]]]:
            for i in range(4):
                yield ('a', i), (2 * i, 3 * i)

        f1 = Frame.from_items(
            gen(),
            name='foo',
            index=(('a', 1), ('a', 2)),
            index_constructor=IndexHierarchy.from_labels,
            columns_constructor=IndexHierarchy.from_labels,
        )
        self.assertEqual(f1.index.__class__, IndexHierarchy)
        self.assertEqual(f1.columns.__class__, IndexHierarchy)

        self.assertEqual(
            f1.to_pairs(),
            (
                (('a', 0), ((('a', 1), 0), (('a', 2), 0))),
                (('a', 1), ((('a', 1), 2), (('a', 2), 3))),
                (('a', 2), ((('a', 1), 4), (('a', 2), 6))),
                (('a', 3), ((('a', 1), 6), (('a', 2), 9))),
            ),
        )

    def test_frame_from_items_h(self) -> None:
        def gen() -> tp.Iterator[tp.Tuple[int, tp.Tuple[int, int]]]:
            for i in range(2):
                yield i, np.array(tuple(str(1000 + j + i) for j in range(3)))

        f1 = Frame.from_items(
            gen(), dtypes=(np.dtype('datetime64[Y]'), np.dtype('datetime64[Y]'))
        )

        self.assertEqual(
            f1.to_pairs(),
            (
                (
                    0,
                    (
                        (0, np.datetime64('1000')),
                        (1, np.datetime64('1001')),
                        (2, np.datetime64('1002')),
                    ),
                ),
                (
                    1,
                    (
                        (0, np.datetime64('1001')),
                        (1, np.datetime64('1002')),
                        (2, np.datetime64('1003')),
                    ),
                ),
            ),
        )

    def test_frame_from_items_i(self) -> None:
        def gen() -> tp.Iterator[tp.Tuple[int, Series]]:
            for i in range(2):
                yield (
                    i,
                    Series(
                        tuple(str(1000 + j + i) for j in range(3)),
                        index=('a', 'b', 'c'),
                    ),
                )

        with self.assertRaises(ErrorInitFrame):
            # must provide an index
            _ = Frame.from_items(
                gen(), dtypes=(np.dtype('datetime64[Y]'), np.dtype('datetime64[Y]'))
            )

        f1 = Frame.from_items(
            gen(),
            dtypes=(np.dtype('datetime64[Y]'), np.dtype('datetime64[Y]')),
            index=('a', 'c'),
        )

        self.assertEqual(
            f1.to_pairs(),
            (
                (0, (('a', np.datetime64('1000')), ('c', np.datetime64('1002')))),
                (1, (('a', np.datetime64('1001')), ('c', np.datetime64('1003')))),
            ),
        )

    def test_frame_from_items_j(self) -> None:
        def gen() -> tp.Iterator[tp.Tuple[int, tp.Tuple[int, int]]]:
            for i in range(2):
                yield i, Frame.from_element('x', index=('a',), columns=('a',))  # type: ignore

        with self.assertRaises(ErrorInitFrame):
            # must provide an index
            _ = Frame.from_items(gen())

    def test_frame_from_items_k1(self) -> None:
        s1 = Series((2, 3), index=('b', 'c'))
        s2 = Series(('x', 'y'), index=('a', 'c'))
        s3 = Series((True, False), index=('b', 'c'))

        f1 = Frame.from_items(
            zip(list('xyz'), (s1, s2, s3)),
            index=tuple('abc'),
            fill_value=FillValueAuto,
        )
        self.assertEqual(
            f1.to_pairs(),
            (
                ('x', (('a', 0), ('b', 2), ('c', 3))),
                ('y', (('a', 'x'), ('b', ''), ('c', 'y'))),
                ('z', (('a', False), ('b', True), ('c', False))),
            ),
        )

    def test_frame_from_items_k2(self) -> None:
        s1 = Series((2, 3), index=('b', 'c'))
        s2 = Series(('x', 'y'), index=('a', 'c'))
        s3 = Series((True, False), index=('b', 'c'))

        f1 = Frame.from_items(
            zip(list('xyz'), (s1, s2, s3)),
            index=tuple('abc'),
            fill_value=defaultdict(lambda: None, {'z': False}),
        )
        self.assertEqual(
            f1.to_pairs(),
            (
                ('x', (('a', None), ('b', 2), ('c', 3))),
                ('y', (('a', 'x'), ('b', None), ('c', 'y'))),
                ('z', (('a', False), ('b', True), ('c', False))),
            ),
        )

    def test_frame_from_items_l(self) -> None:
        s1 = Series((2, 3), index=('b', 'c'))
        s2 = Series(('x', 'y'), index=('a', 'c'))
        s3 = Series((True, False), index=('b', 'c'))

        f1 = Frame.from_items(
            zip(list('xyz'), (s1, s2, s3)),
            index=tuple('abc'),
            fill_value=FillValueAuto,
            dtypes=str,
        )
        self.assertEqual(
            f1.to_pairs(),
            (
                ('x', (('a', ''), ('b', '2'), ('c', '3'))),
                ('y', (('a', 'x'), ('b', ''), ('c', 'y'))),
                ('z', (('a', ''), ('b', 'True'), ('c', 'False'))),
            ),
        )

    # ---------------------------------------------------------------------------

    def test_frame_from_structured_array_a(self) -> None:
        a = np.array(
            [('Venus', 4.87, 464), ('Neptune', 102, -200)],
            dtype=[('name', object), ('mass', 'f4'), ('temperature', 'i4')],
        )

        f = sf.Frame.from_structured_array(
            a, index_depth=1, index_column_first='name', name='foo'
        )

        self.assertEqual(f.shape, (2, 2))
        self.assertEqual(f.name, 'foo')
        self.assertEqual(f['temperature'].sum(), 264)

    def test_frame_from_structured_array_b(self) -> None:
        a = np.array(
            [('Venus', 4.87, 464), ('Neptune', 102, -200)],
            dtype=[('name', object), ('mass', 'f4'), ('temperature', 'i4')],
        )

        f = sf.Frame.from_structured_array(
            a, index_column_first=2, index_depth=1, name='foo'
        )
        self.assertEqual(f['name'].to_pairs(), ((464, 'Venus'), (-200, 'Neptune')))

    def test_frame_from_structured_array_c(self) -> None:
        a = np.array(
            [('Venus', 4.87, 464), ('Neptune', 102, -200)],
            dtype=[('name', 'U7'), ('mass', 'f4'), ('temperature', 'i4')],
        )
        f = sf.Frame.from_structured_array(
            a,
            index_depth=1,
        )
        self.assertEqual(f.shape, (2, 2))
        self.assertEqual(f.index.values.tolist(), ['Venus', 'Neptune'])

    # ---------------------------------------------------------------------------

    def test_frame_sort_index_a(self) -> None:
        # reindex both axis
        records = (
            (2, 2, 'a', False, False),
            (30, 34, 'b', True, False),
            (2, 95, 'c', False, False),
            (30, 73, 'd', True, True),
        )

        f1 = Frame.from_records(
            records,
            columns=('p', 'q', 'r', 's', 't'),
            index=('z', 'x', 'w', 'y'),
            name='foo',
        )

        f2 = f1.sort_index()
        self.assertEqual(
            f2.to_pairs(),
            (
                ('p', (('w', 2), ('x', 30), ('y', 30), ('z', 2))),
                ('q', (('w', 95), ('x', 34), ('y', 73), ('z', 2))),
                ('r', (('w', 'c'), ('x', 'b'), ('y', 'd'), ('z', 'a'))),
                ('s', (('w', False), ('x', True), ('y', True), ('z', False))),
                ('t', (('w', False), ('x', False), ('y', True), ('z', False))),
            ),
        )
        self.assertEqual(f1.name, f2.name)

        self.assertEqual(
            f1.sort_index(ascending=False).to_pairs(),
            (
                ('p', (('z', 2), ('y', 30), ('x', 30), ('w', 2))),
                ('q', (('z', 2), ('y', 73), ('x', 34), ('w', 95))),
                ('r', (('z', 'a'), ('y', 'd'), ('x', 'b'), ('w', 'c'))),
                ('s', (('z', False), ('y', True), ('x', True), ('w', False))),
                ('t', (('z', False), ('y', True), ('x', False), ('w', False))),
            ),
        )

    def test_frame_sort_index_b(self) -> None:
        # reindex both axis
        records = (
            ('a', False, False),
            ('b', True, False),
            ('c', False, False),
            ('d', True, True),
        )

        f1 = Frame.from_records(
            records,
            columns=('p', 'q', 'r'),
            index=IndexHierarchy.from_product((1, 2), (10, 20), name='foo'),
        )

        post = f1.sort_index(ascending=False)

        self.assertEqual(post.index.name, f1.index.name)
        self.assertEqual(
            post.to_pairs(),
            (
                ('p', (((2, 20), 'd'), ((2, 10), 'c'), ((1, 20), 'b'), ((1, 10), 'a'))),
                (
                    'q',
                    (
                        ((2, 20), True),
                        ((2, 10), False),
                        ((1, 20), True),
                        ((1, 10), False),
                    ),
                ),
                (
                    'r',
                    (
                        ((2, 20), True),
                        ((2, 10), False),
                        ((1, 20), False),
                        ((1, 10), False),
                    ),
                ),
            ),
        )

    def test_frame_sort_index_c(self) -> None:
        f1 = ff.parse('s(6,2)|v(int)|i(I,str)')
        f2 = f1.sort_index(key=lambda i: np.array([label[-1].lower() for label in i]))

        self.assertEqual(
            f2.index.values.tolist(), ['zmVj', 'z2Oo', 'zZbu', 'ztsv', 'zUvW', 'zkuW']
        )

    def test_frame_sort_index_d(self) -> None:
        ih1 = IndexHierarchy.from_product(('a', 'b'), (1, 5, 3, -4), ('y', 'z', 'x'))

        f1 = Frame.from_elements(range(len(ih1)), index=ih1, columns=('a', 'b'))

        self.assertEqual(
            f1.sort_index(ascending=(False, False, True)).to_pairs(),
            (
                (
                    'a',
                    (
                        (('b', 5, 'x'), 17),
                        (('b', 5, 'y'), 15),
                        (('b', 5, 'z'), 16),
                        (('b', 3, 'x'), 20),
                        (('b', 3, 'y'), 18),
                        (('b', 3, 'z'), 19),
                        (('b', 1, 'x'), 14),
                        (('b', 1, 'y'), 12),
                        (('b', 1, 'z'), 13),
                        (('b', -4, 'x'), 23),
                        (('b', -4, 'y'), 21),
                        (('b', -4, 'z'), 22),
                        (('a', 5, 'x'), 5),
                        (('a', 5, 'y'), 3),
                        (('a', 5, 'z'), 4),
                        (('a', 3, 'x'), 8),
                        (('a', 3, 'y'), 6),
                        (('a', 3, 'z'), 7),
                        (('a', 1, 'x'), 2),
                        (('a', 1, 'y'), 0),
                        (('a', 1, 'z'), 1),
                        (('a', -4, 'x'), 11),
                        (('a', -4, 'y'), 9),
                        (('a', -4, 'z'), 10),
                    ),
                ),
                (
                    'b',
                    (
                        (('b', 5, 'x'), 17),
                        (('b', 5, 'y'), 15),
                        (('b', 5, 'z'), 16),
                        (('b', 3, 'x'), 20),
                        (('b', 3, 'y'), 18),
                        (('b', 3, 'z'), 19),
                        (('b', 1, 'x'), 14),
                        (('b', 1, 'y'), 12),
                        (('b', 1, 'z'), 13),
                        (('b', -4, 'x'), 23),
                        (('b', -4, 'y'), 21),
                        (('b', -4, 'z'), 22),
                        (('a', 5, 'x'), 5),
                        (('a', 5, 'y'), 3),
                        (('a', 5, 'z'), 4),
                        (('a', 3, 'x'), 8),
                        (('a', 3, 'y'), 6),
                        (('a', 3, 'z'), 7),
                        (('a', 1, 'x'), 2),
                        (('a', 1, 'y'), 0),
                        (('a', 1, 'z'), 1),
                        (('a', -4, 'x'), 11),
                        (('a', -4, 'y'), 9),
                        (('a', -4, 'z'), 10),
                    ),
                ),
            ),
        )

    def test_frame_sort_index_e(self) -> None:
        f1 = Frame(index=IndexHierarchy.from_labels((), depth_reference=2)).sort_index()
        self.assertEqual(f1.shape, (0, 0))
        self.assertEqual(f1.index.shape, (0, 2))

    # ---------------------------------------------------------------------------

    def test_frame_sort_columns_a(self) -> None:
        # reindex both axis
        records = (
            (2, 2, 'a', False, False),
            (30, 34, 'b', True, False),
            (2, 95, 'c', False, False),
            (30, 73, 'd', True, True),
        )

        f1 = Frame.from_records(
            records,
            columns=('t', 's', 'r', 'q', 'p'),
            index=('z', 'x', 'w', 'y'),
            name='foo',
        )

        f2 = f1.sort_columns()
        self.assertEqual(
            f2.to_pairs(),
            (
                ('p', (('z', False), ('x', False), ('w', False), ('y', True))),
                ('q', (('z', False), ('x', True), ('w', False), ('y', True))),
                ('r', (('z', 'a'), ('x', 'b'), ('w', 'c'), ('y', 'd'))),
                ('s', (('z', 2), ('x', 34), ('w', 95), ('y', 73))),
                ('t', (('z', 2), ('x', 30), ('w', 2), ('y', 30))),
            ),
        )

        self.assertEqual(f2.name, f1.name)

    def test_frame_sort_columns_b(self) -> None:
        # reindex both axis
        records = (
            (2, 2, False, False),
            (30, 34, True, False),
            (2, 95, False, False),
            (30, 73, True, True),
        )

        f1 = Frame.from_records(
            records,
            columns=IndexHierarchy.from_product((1, 2), (10, 20), name='foo'),
            index=('z', 'x', 'w', 'y'),
        )

        f2 = f1.sort_columns(ascending=False)

        self.assertEqual(f2.columns.name, f1.columns.name)

        self.assertEqual(
            f2.to_pairs(),
            (
                ((2, 20), (('z', False), ('x', False), ('w', False), ('y', True))),
                ((2, 10), (('z', False), ('x', True), ('w', False), ('y', True))),
                ((1, 20), (('z', 2), ('x', 34), ('w', 95), ('y', 73))),
                ((1, 10), (('z', 2), ('x', 30), ('w', 2), ('y', 30))),
            ),
        )

    def test_frame_sort_columns_c(self) -> None:
        # reindex both axis
        records = (
            (2, 2, False, False),
            (30, 34, True, False),
            (2, 95, False, False),
            (30, 73, True, True),
        )

        f1 = Frame.from_records(
            records,
            columns=IndexYearMonth.from_year_month_range('2018-01', '2018-04'),
            index=('z', 'x', 'w', 'y'),
        )
        f2 = f1.sort_columns(ascending=False)

        self.assertEqual(
            f2.to_pairs(),
            (
                (
                    np.datetime64('2018-04'),
                    (('z', False), ('x', False), ('w', False), ('y', True)),
                ),
                (
                    np.datetime64('2018-03'),
                    (('z', False), ('x', True), ('w', False), ('y', True)),
                ),
                (np.datetime64('2018-02'), (('z', 2), ('x', 34), ('w', 95), ('y', 73))),
                (np.datetime64('2018-01'), (('z', 2), ('x', 30), ('w', 2), ('y', 30))),
            ),
        )

        self.assertEqual(f2.columns.__class__, IndexYearMonth)

    def test_frame_sort_columns_d(self) -> None:
        f1 = ff.parse('s(2,6)|v(int)|c(I,str)')
        f2 = f1.sort_columns(key=lambda i: np.array([label[-1].lower() for label in i]))

        self.assertEqual(
            f2.columns.values.tolist(), ['zmVj', 'z2Oo', 'zZbu', 'ztsv', 'zUvW', 'zkuW']
        )

    def test_frame_sort_columns_e(self) -> None:
        ih1 = IndexHierarchy.from_product(('a', 'b'), (1, 5, 3, -4), ('y', 'z', 'x'))

        f1 = Frame(
            np.arange(2 * len(ih1)).reshape(2, len(ih1)), columns=ih1, index=('a', 'b')
        ).sort_columns(ascending=(False, False, True))

        self.assertEqual(
            f1.sort_columns(ascending=(False, False, True)).to_pairs(),
            (
                (('b', 5, 'x'), (('a', 17), ('b', 41))),
                (('b', 5, 'y'), (('a', 15), ('b', 39))),
                (('b', 5, 'z'), (('a', 16), ('b', 40))),
                (('b', 3, 'x'), (('a', 20), ('b', 44))),
                (('b', 3, 'y'), (('a', 18), ('b', 42))),
                (('b', 3, 'z'), (('a', 19), ('b', 43))),
                (('b', 1, 'x'), (('a', 14), ('b', 38))),
                (('b', 1, 'y'), (('a', 12), ('b', 36))),
                (('b', 1, 'z'), (('a', 13), ('b', 37))),
                (('b', -4, 'x'), (('a', 23), ('b', 47))),
                (('b', -4, 'y'), (('a', 21), ('b', 45))),
                (('b', -4, 'z'), (('a', 22), ('b', 46))),
                (('a', 5, 'x'), (('a', 5), ('b', 29))),
                (('a', 5, 'y'), (('a', 3), ('b', 27))),
                (('a', 5, 'z'), (('a', 4), ('b', 28))),
                (('a', 3, 'x'), (('a', 8), ('b', 32))),
                (('a', 3, 'y'), (('a', 6), ('b', 30))),
                (('a', 3, 'z'), (('a', 7), ('b', 31))),
                (('a', 1, 'x'), (('a', 2), ('b', 26))),
                (('a', 1, 'y'), (('a', 0), ('b', 24))),
                (('a', 1, 'z'), (('a', 1), ('b', 25))),
                (('a', -4, 'x'), (('a', 11), ('b', 35))),
                (('a', -4, 'y'), (('a', 9), ('b', 33))),
                (('a', -4, 'z'), (('a', 10), ('b', 34))),
            ),
        )

    # ---------------------------------------------------------------------------

    def test_frame_sort_values_a(self) -> None:
        # reindex both axis
        records = (
            (2, 2, 'c', False, False),
            (30, 34, 'd', True, False),
            (2, 95, 'a', False, False),
            (30, 73, 'b', True, True),
        )

        f1 = Frame.from_records(
            records,
            columns=('p', 'r', 'q', 't', 's'),
            index=('z', 'x', 'w', 'y'),
            name='foo',
        )

        post = f1.sort_values('q')
        self.assertEqual(post.name, f1.name)

        self.assertEqual(
            post.to_pairs(),
            (
                ('p', (('w', 2), ('y', 30), ('z', 2), ('x', 30))),
                ('r', (('w', 95), ('y', 73), ('z', 2), ('x', 34))),
                ('q', (('w', 'a'), ('y', 'b'), ('z', 'c'), ('x', 'd'))),
                ('t', (('w', False), ('y', True), ('z', False), ('x', True))),
                ('s', (('w', False), ('y', True), ('z', False), ('x', False))),
            ),
        )

        self.assertEqual(
            f1.sort_values('p').to_pairs(),
            (
                ('p', (('z', 2), ('w', 2), ('x', 30), ('y', 30))),
                ('r', (('z', 2), ('w', 95), ('x', 34), ('y', 73))),
                ('q', (('z', 'c'), ('w', 'a'), ('x', 'd'), ('y', 'b'))),
                ('t', (('z', False), ('w', False), ('x', True), ('y', True))),
                ('s', (('z', False), ('w', False), ('x', False), ('y', True))),
            ),
        )

    def test_frame_sort_values_b(self) -> None:
        # reindex both axis
        records = (
            (2, 2, 'c', False, False),
            (30, 34, 'd', True, False),
            (2, 95, 'a', True, False),
            (30, 73, 'b', False, True),
        )

        f1 = Frame.from_records(
            records, columns=('p', 'r', 'q', 't', 's'), index=('z', 'x', 'w', 'y')
        )

        post = f1.sort_values(['p', 't'])

        self.assertEqual(
            post.to_pairs(),
            (
                ('p', (('z', 2), ('w', 2), ('y', 30), ('x', 30))),
                ('r', (('z', 2), ('w', 95), ('y', 73), ('x', 34))),
                ('q', (('z', 'c'), ('w', 'a'), ('y', 'b'), ('x', 'd'))),
                ('t', (('z', False), ('w', True), ('y', False), ('x', True))),
                ('s', (('z', False), ('w', False), ('y', True), ('x', False))),
            ),
        )

    def test_frame_sort_values_c(self) -> None:
        records = (
            (2, 2, 3.5),
            (30, 34, 60.2),
            (2, 95, 1.2),
            (30, 73, 50.2),
        )
        f1 = Frame.from_records(
            records, columns=('p', 'q', 'r'), index=('w', 'x', 'y', 'z'), name='foo'
        )

        f2 = f1.sort_values('y', axis=0)
        self.assertEqual(
            f2.to_pairs(),
            (
                ('r', (('w', 3.5), ('x', 60.2), ('y', 1.2), ('z', 50.2))),
                ('p', (('w', 2), ('x', 30), ('y', 2), ('z', 30))),
                ('q', (('w', 2), ('x', 34), ('y', 95), ('z', 73))),
            ),
        )

        self.assertEqual(f2.name, 'foo')

    def test_frame_sort_values_d(self) -> None:
        a1 = np.arange(8).reshape(4, 2) / 10
        match = (
            ('a', ((0.6, 0.6), (0.4, 0.4), (0.2, 0.2), (0.0, 0.0))),
            ('b', ((0.6, 0.7), (0.4, 0.5), (0.2, 0.3), (0.0, 0.1))),
        )

        f1 = Frame(a1, columns=('a', 'b'))
        f1 = f1.set_index('a')
        f1 = f1.sort_values('b', ascending=False)
        self.assertEqual(f1.to_pairs(), match)
        self.assertEqual(f1.index.name, 'a')

        f2 = FrameGO(a1, columns=('a', 'b'))
        f2 = f2.set_index('a')  # type: ignore
        f2 = f2.sort_values('b', ascending=False)
        self.assertEqual(f2.to_pairs(), match)
        self.assertEqual(f2.index.name, 'a')

    def test_frame_sort_values_e(self) -> None:
        # Ensure index sorting works on internally homogenous frames
        data = np.array([[3, 7, 3], [8, 1, 4], [2, 9, 6]])
        f1 = sf.Frame(data, columns=tuple('abc'), index=tuple('xyz'))
        assert len(f1._blocks._blocks) == 1, 'f1 must be internally homogenous.'

        f1_sorted = f1.sort_values('x', axis=0)

        expected1 = (
            ('x', (('a', 3), ('c', 3), ('b', 7))),
            ('y', (('a', 8), ('c', 4), ('b', 1))),
            ('z', (('a', 2), ('c', 6), ('b', 9))),
        )
        self.assertEqual(expected1, f1_sorted.to_pairs(axis=1))

        # Ensure index sorting works on internally heterogeneous frames
        records = ((4, 2, 3), (2, 3.1, False), (6, False, 3.4))
        f2 = sf.Frame.from_records(
            records,
            columns=tuple('abc'),
            index=tuple('xyz'),
            dtypes=(object, object, object),
        )

        assert len(f2._blocks._blocks) > 1, 'f2 must be internally heterogeneous.'
        f2_sorted = f2.sort_values('x', axis=0)

        expected2 = (
            ('x', (('b', 2), ('c', 3), ('a', 4))),
            ('y', (('b', 3.1), ('c', False), ('a', 2))),
            ('z', (('b', False), ('c', 3.4), ('a', 6))),
        )
        self.assertEqual(expected2, f2_sorted.to_pairs(axis=1))

    def test_frame_sort_values_f(self) -> None:
        # Ensure index sorting works on internally homogenous frames
        data = np.array([[3, 7, 3], [8, 1, 4], [2, 9, 6]])
        f1 = sf.Frame(data, columns=tuple('abc'), index=tuple('xyz'))

        with self.assertRaises(AxisInvalid):
            _ = f1.sort_values(('x', 'z'), axis=-1)

        with self.assertRaises(KeyError):
            _ = f1.sort_values(('x', 'z'), axis=0)

        f2_sorted = f1.sort_values(['x', 'z'], axis=0)
        self.assertEqual(
            f2_sorted.to_pairs(),
            (
                ('a', (('x', 3), ('y', 8), ('z', 2))),
                ('c', (('x', 3), ('y', 4), ('z', 6))),
                ('b', (('x', 7), ('y', 1), ('z', 9))),
            ),
        )

        f3_sorted = f1.sort_values(list(k for k in 'xz'), axis=0)
        self.assertEqual(
            f3_sorted.to_pairs(),
            (
                ('a', (('x', 3), ('y', 8), ('z', 2))),
                ('c', (('x', 3), ('y', 4), ('z', 6))),
                ('b', (('x', 7), ('y', 1), ('z', 9))),
            ),
        )

    def test_frame_sort_values_g(self) -> None:
        records = (
            (2, 2, 3.5),
            (30, 34, 60.2),
            (2, 95, 1.2),
            (30, 73, 50.2),
        )
        f1 = Frame.from_records(
            records, columns=('p', 'q', 'r'), index=('w', 'x', 'y', 'z'), name='foo'
        )

        self.assertEqual(
            f1.sort_values('q', key=lambda s: -s).to_pairs(),
            (
                ('p', (('y', 2), ('z', 30), ('x', 30), ('w', 2))),
                ('q', (('y', 95), ('z', 73), ('x', 34), ('w', 2))),
                ('r', (('y', 1.2), ('z', 50.2), ('x', 60.2), ('w', 3.5))),
            ),
        )

        self.assertEqual(
            f1.sort_values('q', key=lambda s: -s.values).to_pairs(),
            (
                ('p', (('y', 2), ('z', 30), ('x', 30), ('w', 2))),
                ('q', (('y', 95), ('z', 73), ('x', 34), ('w', 2))),
                ('r', (('y', 1.2), ('z', 50.2), ('x', 60.2), ('w', 3.5))),
            ),
        )

        self.assertEqual(
            f1.sort_values('z', axis=0, key=lambda s: -s).to_pairs(),
            (
                ('q', (('w', 2), ('x', 34), ('y', 95), ('z', 73))),
                ('r', (('w', 3.5), ('x', 60.2), ('y', 1.2), ('z', 50.2))),
                ('p', (('w', 2), ('x', 30), ('y', 2), ('z', 30))),
            ),
        )

        self.assertEqual(
            f1.sort_values('z', axis=0, key=lambda s: -s.values).to_pairs(),
            (
                ('q', (('w', 2), ('x', 34), ('y', 95), ('z', 73))),
                ('r', (('w', 3.5), ('x', 60.2), ('y', 1.2), ('z', 50.2))),
                ('p', (('w', 2), ('x', 30), ('y', 2), ('z', 30))),
            ),
        )

    def test_frame_sort_values_h(self) -> None:
        records = (
            (2, 2, 3.5),
            (30, 34, 60.2),
            (2, 95, 1.2),
            (30, 73, 50.2),
        )
        f1 = Frame.from_records(
            records, columns=('p', 'q', 'r'), index=('w', 'x', 'y', 'z'), name='foo'
        )

        with self.assertRaises(RuntimeError):
            _ = f1.sort_values(['r', 'p'], key=lambda f: f.mean().values)

        self.assertEqual(
            f1.sort_values(['r', 'p'], axis=1, key=lambda f: f.mean(axis=1)).to_pairs(),
            (
                ('p', (('y', 2), ('w', 2), ('z', 30), ('x', 30))),
                ('q', (('y', 95), ('w', 2), ('z', 73), ('x', 34))),
                ('r', (('y', 1.2), ('w', 3.5), ('z', 50.2), ('x', 60.2))),
            ),
        )

        self.assertEqual(
            f1.sort_values(
                ['r', 'p'], axis=1, key=lambda f: f.mean(axis=1).values
            ).to_pairs(),
            (
                ('p', (('y', 2), ('w', 2), ('z', 30), ('x', 30))),
                ('q', (('y', 95), ('w', 2), ('z', 73), ('x', 34))),
                ('r', (('y', 1.2), ('w', 3.5), ('z', 50.2), ('x', 60.2))),
            ),
        )

        self.assertEqual(
            f1.sort_values(['r', 'p'], axis=1, key=lambda f: -f).to_pairs(),
            (
                ('p', (('x', 30), ('z', 30), ('w', 2), ('y', 2))),
                ('q', (('x', 34), ('z', 73), ('w', 2), ('y', 95))),
                ('r', (('x', 60.2), ('z', 50.2), ('w', 3.5), ('y', 1.2))),
            ),
        )

        self.assertEqual(
            f1.sort_values(['r', 'p'], axis=1, key=lambda f: -f.values).to_pairs(),
            (
                ('p', (('x', 30), ('z', 30), ('w', 2), ('y', 2))),
                ('q', (('x', 34), ('z', 73), ('w', 2), ('y', 95))),
                ('r', (('x', 60.2), ('z', 50.2), ('w', 3.5), ('y', 1.2))),
            ),
        )

    def test_frame_sort_values_i(self) -> None:
        records = (
            (2, 2, 3.5),
            (30, 34, 60.2),
            (2, 95, 1.2),
            (30, 73, 50.2),
        )
        f1 = Frame.from_records(
            records, columns=('p', 'q', 'r'), index=('w', 'x', 'y', 'z'), name='foo'
        )

        with self.assertRaises(RuntimeError):
            _ = f1.sort_values(['w', 'z'], axis=0, key=lambda f: f.mean(axis=1).values)

        self.assertEqual(
            f1.sort_values(['z', 'w'], axis=0, key=lambda f: -f).to_pairs(),
            (
                ('q', (('w', 2), ('x', 34), ('y', 95), ('z', 73))),
                ('r', (('w', 3.5), ('x', 60.2), ('y', 1.2), ('z', 50.2))),
                ('p', (('w', 2), ('x', 30), ('y', 2), ('z', 30))),
            ),
        )

        self.assertEqual(
            f1.sort_values(['z', 'w'], axis=0, key=lambda f: -f.values).to_pairs(),
            (
                ('q', (('w', 2), ('x', 34), ('y', 95), ('z', 73))),
                ('r', (('w', 3.5), ('x', 60.2), ('y', 1.2), ('z', 50.2))),
                ('p', (('w', 2), ('x', 30), ('y', 2), ('z', 30))),
            ),
        )

    def test_frame_sort_values_j(self) -> None:
        records = (
            (2, 9, 3),
            (8, 7, 6),
            (8, 8, 1),
            (8, 0, 5),
        )
        f1 = Frame.from_records(
            records, columns=('p', 'q', 'r'), index=('w', 'x', 'y', 'z'), name='foo'
        )

        f2 = f1.sort_values(['p', 'q'], ascending=False)

        self.assertEqual(
            f2.to_pairs(),
            (
                ('p', (('y', 8), ('x', 8), ('z', 8), ('w', 2))),
                ('q', (('y', 8), ('x', 7), ('z', 0), ('w', 9))),
                ('r', (('y', 1), ('x', 6), ('z', 5), ('w', 3))),
            ),
        )

        with self.assertRaises(RuntimeError):
            _ = f1.sort_values(['p', 'q'], ascending=(False, True, False))

        with self.assertRaises(RuntimeError):
            _ = f1.sort_values('q', ascending=(False, True, False))

    def test_frame_sort_values_k(self) -> None:
        records = (
            (0, 9, 3),
            (9, 9, 3),
            (2, 7, 6),
            (6, 8, 1),
            (1, 0, 5),
        )
        f1 = Frame.from_records(records, columns=('p', 'q', 'r'), name='foo')

        f2 = f1.sort_values('p', ascending=False)
        self.assertEqual(
            f2.to_pairs(),
            (
                ('p', ((1, 9), (3, 6), (2, 2), (4, 1), (0, 0))),
                ('q', ((1, 9), (3, 8), (2, 7), (4, 0), (0, 9))),
                ('r', ((1, 3), (3, 1), (2, 6), (4, 5), (0, 3))),
            ),
        )

    def test_frame_sort_values_m(self) -> None:
        records = (
            (1, 3, 3),
            (9, 8, 3),
            (9, 7, 6),
            (9, 8, 1),
            (0, 0, 5),
        )
        f1 = Frame.from_records(records, columns=('p', 'q', 'r'), name='foo')

        f2 = f1.sort_values(['p', 'q', 'r'], ascending=(True, False, True))
        self.assertEqual(
            f2.to_pairs(),
            (
                ('p', ((4, 0), (0, 1), (3, 9), (1, 9), (2, 9))),
                ('q', ((4, 0), (0, 3), (3, 8), (1, 8), (2, 7))),
                ('r', ((4, 5), (0, 3), (3, 1), (1, 3), (2, 6))),
            ),
        )

        f3 = f1.sort_values(['p', 'q', 'r'], ascending=(False, True, False))
        self.assertEqual(
            f3.to_pairs(),
            (
                ('p', ((2, 9), (1, 9), (3, 9), (0, 1), (4, 0))),
                ('q', ((2, 7), (1, 8), (3, 8), (0, 3), (4, 0))),
                ('r', ((2, 6), (1, 3), (3, 1), (0, 3), (4, 5))),
            ),
        )

        f4 = f1.sort_values(['p', 'q', 'r'], ascending=(True, False, False))
        self.assertEqual(
            f4.to_pairs(),
            (
                ('p', ((4, 0), (0, 1), (1, 9), (3, 9), (2, 9))),
                ('q', ((4, 0), (0, 3), (1, 8), (3, 8), (2, 7))),
                ('r', ((4, 5), (0, 3), (1, 3), (3, 1), (2, 6))),
            ),
        )

    def test_frame_sort_values_n(self) -> None:
        # Ensure index sorting works on internally homogenous frames
        data = np.array([[3, 7, 3], [8, 1, 4], [2, 9, 6]])
        f1 = sf.Frame(data, columns=tuple('abc'), index=tuple('xyz'))
        f2 = f1.sort_values(slice('z', None), axis=0, key=lambda f: -f)  # type: ignore
        self.assertEqual(
            f2.to_pairs(),
            (
                ('b', (('x', 7), ('y', 1), ('z', 9))),
                ('c', (('x', 3), ('y', 4), ('z', 6))),
                ('a', (('x', 3), ('y', 8), ('z', 2))),
            ),
        )

    def test_frame_sort_values_o(self) -> None:
        # Ensure index sorting works on internally homogenous frames
        data = np.array([[3, 7, 3], [8, 1, 4], [2, 9, 6]])
        f1 = sf.Frame(data, columns=tuple('abc'), index=tuple('xyz'))
        f2 = f1.sort_values(slice(None, 'a'), axis=1, key=lambda f: -f.values)
        self.assertEqual(
            f2.to_pairs(),
            (
                ('a', (('y', 8), ('x', 3), ('z', 2))),
                ('b', (('y', 1), ('x', 7), ('z', 9))),
                ('c', (('y', 4), ('x', 3), ('z', 6))),
            ),
        )

    def test_frame_sort_values_p(self) -> None:
        data = np.array([[3, 7, 3], [8, 1, 4], [2, 9, 6]])
        f1 = sf.Frame(data, columns=tuple('abc'), index=tuple('xyz'))
        f2 = f1.sort_values(slice(None, 'x'), axis=0)
        self.assertEqual(
            f2.to_pairs(),
            (
                ('a', (('x', 3), ('y', 8), ('z', 2))),
                ('c', (('x', 3), ('y', 4), ('z', 6))),
                ('b', (('x', 7), ('y', 1), ('z', 9))),
            ),
        )

    def test_frame_sort_values_q(self) -> None:
        f1 = ff.parse('s(2,4)')
        self.assertEqual(f1.sort_values(0, axis=1).index.values.tolist(), [1, 0])

    # ---------------------------------------------------------------------------

    def test_frame_relabel_a(self) -> None:
        # reindex both axis
        records = (
            (2, 2, 'c', False, False),
            (30, 34, 'd', True, False),
            (2, 95, 'a', False, False),
            (30, 73, 'b', True, True),
        )

        f1 = FrameGO.from_records(
            records, columns=('p', 'r', 'q', 't', 's'), index=('z', 'x', 'w', 'y')
        )

        f2 = f1.relabel(columns={'q': 'QQQ'})

        self.assertEqual(
            f2.to_pairs(),
            (
                ('p', (('z', 2), ('x', 30), ('w', 2), ('y', 30))),
                ('r', (('z', 2), ('x', 34), ('w', 95), ('y', 73))),
                ('QQQ', (('z', 'c'), ('x', 'd'), ('w', 'a'), ('y', 'b'))),
                ('t', (('z', False), ('x', True), ('w', False), ('y', True))),
                ('s', (('z', False), ('x', False), ('w', False), ('y', True))),
            ),
        )

        f3 = f1.relabel(index={'y': 'YYY'})

        self.assertEqual(
            f3.to_pairs(),
            (
                ('p', (('z', 2), ('x', 30), ('w', 2), ('YYY', 30))),
                ('r', (('z', 2), ('x', 34), ('w', 95), ('YYY', 73))),
                ('q', (('z', 'c'), ('x', 'd'), ('w', 'a'), ('YYY', 'b'))),
                ('t', (('z', False), ('x', True), ('w', False), ('YYY', True))),
                ('s', (('z', False), ('x', False), ('w', False), ('YYY', True))),
            ),
        )

        self.assertTrue((f1.mloc == f2.mloc).all())
        self.assertTrue((f2.mloc == f3.mloc).all())

    def test_frame_relabel_b(self) -> None:
        # reindex both axis
        records = (
            (2, 2, 'c', False),
            (30, 34, 'd', True),
        )

        f1 = FrameGO.from_records(records, columns=('p', 'r', 'q', 't'), index=('x', 'y'))

        f2 = f1.relabel(columns=IndexAutoFactory)
        self.assertEqual(f2.columns.values.tolist(), [0, 1, 2, 3])

    def test_frame_relabel_c(self) -> None:
        records = (
            (1, 2, 'a', False),
            (30, 34, 'b', True),
        )
        f1 = Frame.from_records(
            records, columns=tuple('pqrs'), index=tuple('ab'), name='foo'
        )

        f2 = f1.relabel(index=IndexAutoFactory)
        self.assertEqual(
            f2.to_pairs(),
            (
                ('p', ((0, 1), (1, 30))),
                ('q', ((0, 2), (1, 34))),
                ('r', ((0, 'a'), (1, 'b'))),
                ('s', ((0, False), (1, True))),
            ),
        )

        f3 = f1.relabel(columns=IndexAutoFactory)
        self.assertEqual(
            f3.relabel(columns=IndexAutoFactory).to_pairs(),
            (
                (0, (('a', 1), ('b', 30))),
                (1, (('a', 2), ('b', 34))),
                (2, (('a', 'a'), ('b', 'b'))),
                (3, (('a', False), ('b', True))),
            ),
        )
        self.assertTrue(f3.columns.STATIC)

    def test_frame_relabel_d(self) -> None:
        records = (
            (1, 2, 'a', False),
            (30, 34, 'b', True),
        )
        f1 = FrameGO.from_records(
            records, columns=tuple('pqrs'), index=tuple('ab'), name='foo'
        )

        f2 = f1.relabel(columns=IndexAutoFactory)
        self.assertEqual(
            f2.relabel(columns=IndexAutoFactory).to_pairs(),
            (
                (0, (('a', 1), ('b', 30))),
                (1, (('a', 2), ('b', 34))),
                (2, (('a', 'a'), ('b', 'b'))),
                (3, (('a', False), ('b', True))),
            ),
        )
        self.assertFalse(f2.columns.STATIC)
        f2[4] = None
        self.assertTrue(f2.columns._map is None)  # type: ignore
        f2[6] = None
        self.assertFalse(f2.columns._map is None)  # type: ignore

        self.assertEqual(
            f2.to_pairs(),
            (
                (0, (('a', 1), ('b', 30))),
                (1, (('a', 2), ('b', 34))),
                (2, (('a', 'a'), ('b', 'b'))),
                (3, (('a', False), ('b', True))),
                (4, (('a', None), ('b', None))),
                (6, (('a', None), ('b', None))),
            ),
        )

    def test_frame_relabel_e(self) -> None:
        f1 = FrameGO.from_dict(
            {('A', 1): (10, 20), ('A', 2): (40, 50), ('B', 1): (30, 50)}
        )
        # we have to convert the IH to an TVIHGO
        f2 = f1.relabel(columns=IndexHierarchy.from_labels(f1.columns))
        self.assertEqual(f2.columns.__class__, IndexHierarchyGO)
        self.assertEqual(
            f2.to_pairs(),
            (
                (('A', 1), ((0, 10), (1, 20))),
                (('A', 2), ((0, 40), (1, 50))),
                (('B', 1), ((0, 30), (1, 50))),
            ),
        )

    def test_frame_relabel_f(self) -> None:
        records = (
            (2, 'a', False),
            (34, 'b', True),
        )
        f1 = Frame.from_records(records, columns=('p', 'q', 'r'), index=('w', 'x'))
        # will get new instance but reuse index
        f2 = f1.relabel()
        self.assertEqual(id(f2.index), id(f1.index))
        self.assertEqual(id(f2.columns), id(f1.columns))

    def test_frame_relabel_g(self) -> None:
        f1 = FrameGO.from_elements([1, 2], columns=['a'])
        f1['b'] = f1['a']
        f2 = f1.relabel(
            columns={
                'a': 'c',
            }
        )
        self.assertEqual(
            f2.to_pairs(), (('c', ((0, 1), (1, 2))), ('b', ((0, 1), (1, 2))))
        )

    def test_frame_relabel_h(self) -> None:
        f1 = ff.parse('s(2,2)')
        with self.assertRaises(RuntimeError):
            f1.relabel(columns={'a', 'c'})

        with self.assertRaises(RuntimeError):
            f1.relabel(index={'a', 'c'})

    def test_frame_relabel_i(self) -> None:
        f1 = ff.parse('s(2,2)')
        f2 = f1.relabel(index=('1982-01', '1994-05'), index_constructor=IndexYearMonth)
        self.assertEqual(f2.index.__class__, IndexYearMonth)

    def test_frame_relabel_j(self) -> None:
        f1 = ff.parse('s(2,2)').relabel(columns=('1982-01', '1994-05'))
        f2 = f1.relabel(columns_constructor=IndexYearMonth)
        self.assertEqual(f2.columns.__class__, IndexYearMonth)

    # ---------------------------------------------------------------------------

    def test_frame_rehierarch_a(self) -> None:
        records = (
            (1, 2, 'a', False),
            (30, 34, 'b', True),
            (54, 95, 'c', False),
            (65, 73, 'd', True),
        )
        columns = IndexHierarchy.from_product(('a', 'b'), (1, 2))
        index = IndexHierarchy.from_product((100, 200), (True, False))
        f1 = Frame.from_records(records, columns=columns, index=index)

        f2 = f1.rehierarch(index=(1, 0), columns=(1, 0))
        self.assertEqual(
            f2.to_pairs(),
            (
                (
                    (1, 'a'),
                    (
                        ((True, 100), 1),
                        ((True, 200), 54),
                        ((False, 100), 30),
                        ((False, 200), 65),
                    ),
                ),
                (
                    (1, 'b'),
                    (
                        ((True, 100), 'a'),
                        ((True, 200), 'c'),
                        ((False, 100), 'b'),
                        ((False, 200), 'd'),
                    ),
                ),
                (
                    (2, 'a'),
                    (
                        ((True, 100), 2),
                        ((True, 200), 95),
                        ((False, 100), 34),
                        ((False, 200), 73),
                    ),
                ),
                (
                    (2, 'b'),
                    (
                        ((True, 100), False),
                        ((True, 200), False),
                        ((False, 100), True),
                        ((False, 200), True),
                    ),
                ),
            ),
        )

    def test_frame_rehierarch_b(self) -> None:
        records = (
            (1, 2, 'a', False),
            (30, 34, 'b', True),
        )
        f1 = FrameGO.from_records(
            records, columns=tuple('pqrs'), index=tuple('ab'), name='foo'
        )

        # no hierarchy fails
        with self.assertRaises(RuntimeError):
            f1.rehierarch(index=(0, 1))

        with self.assertRaises(RuntimeError):
            f1.rehierarch(columns=(0, 1))

    def test_frame_rehierarch_c(self) -> None:
        records = (
            (1, 2, 'a', False),
            (30, 34, 'b', True),
            (54, 95, 'c', False),
            (65, 73, 'd', True),
        )
        columns = IndexHierarchy.from_product(('a', 'b'), (1, 2))
        index = IndexHierarchy.from_product((100, 200), (True, False))
        f1 = Frame.from_records(records, columns=columns, index=index)

        f2 = f1.rehierarch(index=(1, 0))
        self.assertEqual(
            f2.to_pairs(),
            (
                (
                    ('a', 1),
                    (
                        ((True, 100), 1),
                        ((True, 200), 54),
                        ((False, 100), 30),
                        ((False, 200), 65),
                    ),
                ),
                (
                    ('a', 2),
                    (
                        ((True, 100), 2),
                        ((True, 200), 95),
                        ((False, 100), 34),
                        ((False, 200), 73),
                    ),
                ),
                (
                    ('b', 1),
                    (
                        ((True, 100), 'a'),
                        ((True, 200), 'c'),
                        ((False, 100), 'b'),
                        ((False, 200), 'd'),
                    ),
                ),
                (
                    ('b', 2),
                    (
                        ((True, 100), False),
                        ((True, 200), False),
                        ((False, 100), True),
                        ((False, 200), True),
                    ),
                ),
            ),
        )

        f3 = f1.rehierarch(columns=(1, 0))
        self.assertEqual(
            f3.to_pairs(),
            (
                (
                    (1, 'a'),
                    (
                        ((100, True), 1),
                        ((100, False), 30),
                        ((200, True), 54),
                        ((200, False), 65),
                    ),
                ),
                (
                    (1, 'b'),
                    (
                        ((100, True), 'a'),
                        ((100, False), 'b'),
                        ((200, True), 'c'),
                        ((200, False), 'd'),
                    ),
                ),
                (
                    (2, 'a'),
                    (
                        ((100, True), 2),
                        ((100, False), 34),
                        ((200, True), 95),
                        ((200, False), 73),
                    ),
                ),
                (
                    (2, 'b'),
                    (
                        ((100, True), False),
                        ((100, False), True),
                        ((200, True), False),
                        ((200, False), True),
                    ),
                ),
            ),
        )

    def test_frame_rehierarch_d(self) -> None:
        records = (
            (1, 2, 'a', False),
            (30, 34, 'b', True),
            (54, 95, 'c', False),
            (65, 73, 'd', True),
        )
        columns = IndexHierarchy.from_product(('a', 'b'), (1, 2))
        index = IndexHierarchy.from_product((100, 200), (True, False))
        f1 = FrameGO.from_records(records, columns=columns, index=index)

        f1[('b', 3)] = 'new'

        f3 = f1.rehierarch(columns=(1, 0))
        self.assertEqual(
            f3.to_pairs(),
            (
                (
                    (1, 'a'),
                    (
                        ((100, True), 1),
                        ((100, False), 30),
                        ((200, True), 54),
                        ((200, False), 65),
                    ),
                ),
                (
                    (1, 'b'),
                    (
                        ((100, True), 'a'),
                        ((100, False), 'b'),
                        ((200, True), 'c'),
                        ((200, False), 'd'),
                    ),
                ),
                (
                    (2, 'a'),
                    (
                        ((100, True), 2),
                        ((100, False), 34),
                        ((200, True), 95),
                        ((200, False), 73),
                    ),
                ),
                (
                    (2, 'b'),
                    (
                        ((100, True), False),
                        ((100, False), True),
                        ((200, True), False),
                        ((200, False), True),
                    ),
                ),
                (
                    (3, 'b'),
                    (
                        ((100, True), 'new'),
                        ((100, False), 'new'),
                        ((200, True), 'new'),
                        ((200, False), 'new'),
                    ),
                ),
            ),
        )

    def test_frame_rehierarch_e(self) -> None:
        f = sf.Frame.from_items(
            [('price', [2, 3, 4])],
            index=sf.IndexHierarchy.from_product(
                ['APPL'],
                sf.IndexDate(
                    [
                        np.datetime64('2022-01-01'),
                        np.datetime64('2022-01-02'),
                        np.datetime64('2022-01-03'),
                    ]
                ),
            ),
        )
        f.rehierarch((1, 0))
        self.assertEqual(
            f.to_pairs(),
            (
                (
                    'price',
                    (
                        (('APPL', np.datetime64('2022-01-01')), 2),
                        (('APPL', np.datetime64('2022-01-02')), 3),
                        (('APPL', np.datetime64('2022-01-03')), 4),
                    ),
                ),
            ),
        )

    # ---------------------------------------------------------------------------

    def test_frame_get_a(self) -> None:
        # reindex both axis
        records = (
            (2, 2, 'c', False, False),
            (30, 34, 'd', True, False),
            (2, 95, 'a', False, False),
            (30, 73, 'b', True, True),
        )

        f1 = FrameGO.from_records(
            records, columns=('p', 'r', 'q', 't', 's'), index=('z', 'x', 'w', 'y')
        )

        self.assertEqual(f1.get('r').values.tolist(), [2, 34, 95, 73])

        self.assertEqual(f1.get('a'), None)
        self.assertEqual(f1.get('w'), None)
        self.assertEqual(f1.get('a', -1), -1)

    def test_frame_isna_a(self) -> None:
        f1 = FrameGO.from_records(
            [[np.nan, 2, np.nan, 0], [3, 4, np.nan, 1], [np.nan, np.nan, np.nan, 5]],
            columns=list('ABCD'),
        )

        self.assertEqual(
            f1.isna().to_pairs(),
            (
                ('A', ((0, True), (1, False), (2, True))),
                ('B', ((0, False), (1, False), (2, True))),
                ('C', ((0, True), (1, True), (2, True))),
                ('D', ((0, False), (1, False), (2, False))),
            ),
        )

        self.assertEqual(
            f1.notna().to_pairs(),
            (
                ('A', ((0, False), (1, True), (2, False))),
                ('B', ((0, True), (1, True), (2, False))),
                ('C', ((0, False), (1, False), (2, False))),
                ('D', ((0, True), (1, True), (2, True))),
            ),
        )

    def test_frame_isna_b(self) -> None:
        # f1 will wave 2 blocks, where as f2 will have single contiguous block
        f1 = sf.Frame.from_dict({'a': ['', ''], 'b': ['', '']}, dtypes=('<U7', '<U9'))
        f2 = sf.Frame.from_element('', columns=['a', 'b'], index=[0, 1])

        self.assertEqual(
            f1.isna().to_pairs(),
            (('a', ((0, False), (1, False))), ('b', ((0, False), (1, False)))),
        )
        self.assertEqual(
            f2.isna().to_pairs(),
            (('a', ((0, False), (1, False))), ('b', ((0, False), (1, False)))),
        )

        post1 = f1.isna().sum()
        self.assertEqual(post1.to_pairs(), (('a', 0), ('b', 0)))

        post2 = f2.isna().sum()
        self.assertEqual(post2.to_pairs(), (('a', 0), ('b', 0)))

    def test_frame_isna_c(self) -> None:
        f1 = sf.Frame.from_dict(
            dict(a=(True, True, True), b=(True, True, True)), index=range(3)
        )
        post = f1.sum()
        self.assertEqual(post.to_pairs(), (('a', 3), ('b', 3)))

    # ---------------------------------------------------------------------------

    def test_frame_dropna_a(self) -> None:
        f1 = FrameGO.from_records(
            [
                [np.nan, 2, np.nan, 0],
                [3, 4, np.nan, 1],
                [np.nan, np.nan, np.nan, np.nan],
            ],
            columns=list('ABCD'),
        )

        self.assertAlmostEqualFramePairs(
            f1.dropna(axis=0, condition=np.all).to_pairs(),
            (
                ('A', ((0, nan), (1, 3.0))),
                ('B', ((0, 2.0), (1, 4.0))),
                ('C', ((0, nan), (1, nan))),
                ('D', ((0, 0.0), (1, 1.0))),
            ),
        )

        self.assertAlmostEqualFramePairs(
            f1.dropna(axis=1, condition=np.all).to_pairs(),
            (
                ('A', ((0, nan), (1, 3.0), (2, nan))),
                ('B', ((0, 2.0), (1, 4.0), (2, nan))),
                ('D', ((0, 0.0), (1, 1.0), (2, nan))),
            ),
        )

        f2 = f1.dropna(axis=0, condition=np.any)
        # dropping to zero results in an empty DF in the same manner as Pandas; not sure if this is correct or ideal
        self.assertEqual(f2.shape, (0, 4))

        f3 = f1.dropna(axis=1, condition=np.any)
        self.assertEqual(f3.shape, (3, 0))

    def test_frame_dropna_b(self) -> None:
        f1 = FrameGO.from_records(
            [[np.nan, 2, 3, 0], [3, 4, np.nan, 1], [0, 1, 2, 3]], columns=list('ABCD')
        )

        self.assertEqual(
            f1.dropna(axis=0, condition=np.any).to_pairs(),
            (
                ('A', ((2, 0.0),)),
                ('B', ((2, 1.0),)),
                ('C', ((2, 2.0),)),
                ('D', ((2, 3.0),)),
            ),
        )
        self.assertEqual(
            f1.dropna(axis=1, condition=np.any).to_pairs(),
            (
                ('B', ((0, 2.0), (1, 4.0), (2, 1.0))),
                ('D', ((0, 0.0), (1, 1.0), (2, 3.0))),
            ),
        )

    def test_frame_dropna_c(self) -> None:
        f1 = Frame.from_records(
            [
                [np.nan, np.nan],
                [np.nan, np.nan],
            ],
            columns=list('AB'),
        )
        f2 = f1.dropna()
        self.assertEqual(f2.shape, (0, 2))

    def test_frame_dropna_d(self) -> None:
        f1 = Frame(np.arange(4).reshape(2, 2), columns=list('ab'))
        f2 = f1.dropna()
        self.assertEqual(id(f1), id(f2))

        f3 = FrameGO(np.arange(4).reshape(2, 2), columns=list('ab'))
        f4 = f3.dropna()
        self.assertNotEqual(id(f3), id(f4))

    def test_frame_dropna_e(self) -> None:
        f1 = sf.Series([1, 2]).to_frame()
        f2 = f1.dropna()
        self.assertEqual(f2.to_pairs(), ((0, ((0, 1), (1, 2))),))

        f3 = sf.Series([1, 2, np.nan]).to_frame()
        f4 = f3.dropna()
        self.assertEqual(f4.to_pairs(), ((0, ((0, 1), (1, 2))),))

    # ---------------------------------------------------------------------------

    def test_frame_isfalsy_a(self) -> None:
        f1 = FrameGO.from_records(
            [[False, 2, None, 0], [3, 4, 5, 1], ['', '', '', '']], columns=list('ABCD')
        )

        f2 = f1.isfalsy()
        self.assertEqual(
            f2.to_pairs(),
            (
                ('A', ((0, True), (1, False), (2, True))),
                ('B', ((0, False), (1, False), (2, True))),
                ('C', ((0, True), (1, False), (2, True))),
                ('D', ((0, True), (1, False), (2, True))),
            ),
        )

    # ---------------------------------------------------------------------------

    def test_frame_notfalsy_a(self) -> None:
        f1 = FrameGO.from_records(
            [[False, 2, None, 0], [3, 4, 5, 1], ['', '', '', '']], columns=list('ABCD')
        )

        f2 = f1.notfalsy()
        self.assertEqual(
            f2.to_pairs(),
            (
                ('A', ((0, False), (1, True), (2, False))),
                ('B', ((0, True), (1, True), (2, False))),
                ('C', ((0, False), (1, True), (2, False))),
                ('D', ((0, False), (1, True), (2, False))),
            ),
        )

    # ---------------------------------------------------------------------------

    def test_frame_dropfalsy_a(self) -> None:
        f1 = FrameGO.from_records(
            [[False, 2, None, 0], [3, 4, 5, 1], ['', '', '', '']], columns=list('ABCD')
        )

        self.assertEqual(
            f1.dropfalsy().to_pairs(),
            (
                ('A', ((0, False), (1, 3))),
                ('B', ((0, 2), (1, 4))),
                ('C', ((0, None), (1, 5))),
                ('D', ((0, 0), (1, 1))),
            ),
        )

        self.assertEqual(
            f1.dropfalsy(condition=np.any).to_pairs(),
            (('A', ((1, 3),)), ('B', ((1, 4),)), ('C', ((1, 5),)), ('D', ((1, 1),))),
        )

        self.assertEqual(f1.dropfalsy(axis=1, condition=np.any).shape, (3, 0))

    def test_frame_dropfalsy_b(self) -> None:
        f1 = Frame(np.arange(4).reshape(2, 2), columns=list('ab'))
        f2 = f1.dropfalsy()
        self.assertEqual(id(f1), id(f2))

        f3 = FrameGO(np.arange(4).reshape(2, 2), columns=list('ab'))
        f4 = f3.dropfalsy()
        self.assertNotEqual(id(f3), id(f4))

    # ---------------------------------------------------------------------------

    @skip_win
    def test_frame_fillna_a(self) -> None:
        dtype = np.dtype

        f1 = FrameGO.from_records(
            [[np.nan, 2, 3, 0], [3, 4, np.nan, 1], [0, 1, 2, 3]], columns=list('ABCD')
        )

        f2 = f1.fillna(0)
        self.assertEqual(
            f2.to_pairs(),
            (
                ('A', ((0, 0.0), (1, 3.0), (2, 0.0))),
                ('B', ((0, 2.0), (1, 4.0), (2, 1.0))),
                ('C', ((0, 3.0), (1, 0.0), (2, 2.0))),
                ('D', ((0, 0.0), (1, 1.0), (2, 3.0))),
            ),
        )

        post = f2.dtypes
        self.assertEqual(
            post.to_pairs(),
            (
                ('A', dtype('float64')),
                ('B', dtype('int64')),
                ('C', dtype('float64')),
                ('D', dtype('int64')),
            ),
        )

        f3 = f1.fillna(None)
        self.assertEqual(
            f3.to_pairs(),
            (
                ('A', ((0, None), (1, 3.0), (2, 0.0))),
                ('B', ((0, 2.0), (1, 4.0), (2, 1.0))),
                ('C', ((0, 3.0), (1, None), (2, 2.0))),
                ('D', ((0, 0.0), (1, 1.0), (2, 3.0))),
            ),
        )

        post = f3.dtypes
        self.assertEqual(
            post.to_pairs(),
            (
                ('A', dtype('O')),
                ('B', dtype('int64')),
                ('C', dtype('O')),
                ('D', dtype('int64')),
            ),
        )

    @skip_win
    def test_frame_fillna_b(self) -> None:
        f1 = Frame.from_records(
            [[np.nan, 2, 3, 0], [3, np.nan, 20, 1], [0, 1, 2, 3]],
            columns=tuple('ABCD'),
            index=tuple('wxy'),
        )

        f2 = Frame.from_records(
            [
                [300, 2],
                [3, 200],
            ],
            columns=tuple('AB'),
            index=tuple('wx'),
        )

        f3 = f1.fillna(f2)

        self.assertEqual(
            f3.dtypes.values.tolist(),
            [
                np.dtype('float64'),
                np.dtype('float64'),
                np.dtype('int64'),
                np.dtype('int64'),
            ],
        )
        self.assertEqual(
            f3.to_pairs(),
            (
                ('A', (('w', 300.0), ('x', 3.0), ('y', 0.0))),
                ('B', (('w', 2.0), ('x', 200.0), ('y', 1.0))),
                ('C', (('w', 3), ('x', 20), ('y', 2))),
                ('D', (('w', 0), ('x', 1), ('y', 3))),
            ),
        )

    def test_frame_fillna_c(self) -> None:
        f1 = Frame.from_records(
            [[np.nan, 2, 3, 0], [3, 30, None, None], [0, np.nan, 2, 3]],
            columns=tuple('ABCD'),
            index=tuple('wxy'),
        )

        f2 = Frame.from_records(
            [
                [300, 230],
                [110, 200],
                [580, 750],
            ],
            columns=tuple('AB'),
            index=tuple('yxw'),
        )

        f3 = f1.fillna(f2)

        self.assertEqual(
            f3.dtypes.values.tolist(),
            [np.dtype('float64'), np.dtype('float64'), np.dtype('O'), np.dtype('O')],
        )
        self.assertEqual(
            f3.to_pairs(),
            (
                ('A', (('w', 580.0), ('x', 3.0), ('y', 0.0))),
                ('B', (('w', 2.0), ('x', 30.0), ('y', 230.0))),
                ('C', (('w', 3), ('x', None), ('y', 2))),
                ('D', (('w', 0), ('x', None), ('y', 3))),
            ),
        )

    def test_frame_fillna_d(self) -> None:
        f1 = Frame.from_records(
            [[None, 2, 3, 0], [3, 30, None, None], [0, None, 2, 3]],
            columns=tuple('ABCD'),
            index=tuple('wxy'),
        )

        f2 = Frame.from_records(
            [
                [300, 230],
                [110, 200],
            ],
            columns=tuple('CB'),
            index=tuple('yx'),
        )

        f3 = f1.fillna(f2)

        self.assertEqual(
            f3.to_pairs(),
            (
                ('A', (('w', None), ('x', 3), ('y', 0))),
                ('B', (('w', 2), ('x', 30), ('y', 230))),
                ('C', (('w', 3), ('x', 110), ('y', 2))),
                ('D', (('w', 0), ('x', None), ('y', 3))),
            ),
        )

        # assure we do not fill with float when reindexing
        self.assertEqual([type(v) for v in f3['B'].values.tolist()], [int, int, int])

    def test_frame_fillna_e(self) -> None:
        f = Frame.from_fields(((None, 'foo'), ('a', np.nan)))
        with self.assertRaises(ValueError):
            # ValueError: Fill values must be one-dimensional arrays.
            f.fillna(np.arange(4).reshape(2, 2))

    def test_frame_fillna_f(self) -> None:
        f1 = Frame.from_records(
            [[np.nan, 2, 3, 0], [3, 30, None, None], [0, np.nan, 2, 3]],
            columns=tuple('ABCD'),
            index=tuple('wxy'),
        )
        f2 = f1.fillna({'A': None, 'B': 0, 'C': -1, 'D': None})
        self.assertEqual(
            f2.to_pairs(),
            (
                ('A', (('w', None), ('x', 3.0), ('y', 0.0))),
                ('B', (('w', 2.0), ('x', 30.0), ('y', 0.0))),
                ('C', (('w', 3), ('x', -1), ('y', 2))),
                ('D', (('w', 0), ('x', None), ('y', 3))),
            ),
        )

    def test_frame_fillna_g(self) -> None:
        f1 = Frame.from_records(
            [[np.nan, 2, 3, 0], [3, 30, None, None], [0, np.nan, 2, 3]],
            columns=tuple('ABCD'),
            index=tuple('wxy'),
        )
        f2 = f1.fillna([None, 0, -1, None])
        self.assertEqual(
            f2.to_pairs(),
            (
                ('A', (('w', None), ('x', 3.0), ('y', 0.0))),
                ('B', (('w', 2.0), ('x', 30.0), ('y', 0.0))),
                ('C', (('w', 3), ('x', -1), ('y', 2))),
                ('D', (('w', 0), ('x', None), ('y', 3))),
            ),
        )

    def test_frame_fillna_h(self) -> None:
        f1 = Frame.from_records(
            [[np.nan, 2, 3, 0], [3, 30, None, None], [0, np.nan, 2, 3]],
            columns=tuple('ABCD'),
            index=tuple('wxy'),
        )
        f2 = f1.fillna(FillValueAuto(f=-1, O='na'))
        self.assertEqual(
            f2.to_pairs(),
            (
                ('A', (('w', -1.0), ('x', 3.0), ('y', 0.0))),
                ('B', (('w', 2.0), ('x', 30.0), ('y', -1.0))),
                ('C', (('w', 3), ('x', 'na'), ('y', 2))),
                ('D', (('w', 0), ('x', 'na'), ('y', 3))),
            ),
        )

    def test_frame_fillna_i(self) -> None:
        f1 = sf.Frame.from_fields(
            (('', 'foo'), ('bar', None), ('baz', '')), columns=('a', 'b', 'c')
        )
        f2 = f1.fillna({'a': 'x', 'b': 'x', 'c': 'y'})
        self.assertEqual(
            f2.to_pairs(),
            (
                ('a', ((0, ''), (1, 'foo'))),
                ('b', ((0, 'bar'), (1, 'x'))),
                ('c', ((0, 'baz'), (1, ''))),
            ),
        )

        f3 = f1.fillfalsy({'a': 'x', 'b': 'x', 'c': 'y'})
        self.assertEqual(
            f3.to_pairs(),
            (
                ('a', ((0, 'x'), (1, 'foo'))),
                ('b', ((0, 'bar'), (1, 'x'))),
                ('c', ((0, 'baz'), (1, 'y'))),
            ),
        )

    def test_frame_fillna_j(self) -> None:
        f1 = sf.Frame.from_fields(
            ((1.1, 2.2), (3.1, np.nan), ('bar', 'foo'), ('baz', None)),
            columns=('a', 'b', 'c', 'd'),
            consolidate_blocks=True,
        )

        f2 = f1.fillna({'b': 'x', 'd': 'y'})
        self.assertEqual(
            f2.to_pairs(),
            (
                ('a', ((0, 1.1), (1, 2.2))),
                ('b', ((0, 3.1), (1, 'x'))),
                ('c', ((0, 'bar'), (1, 'foo'))),
                ('d', ((0, 'baz'), (1, 'y'))),
            ),
        )

    # ---------------------------------------------------------------------------

    def test_frame_fillfalsy_a(self) -> None:
        f1 = Frame.from_records(
            [[None, 2, 3, 0], [3, 30, '', False], [0, np.nan, 2, 3]],
            columns=tuple('ABCD'),
            index=tuple('wxy'),
        )

        self.assertEqual(
            f1.fillfalsy(-1).to_pairs(),
            (
                ('A', (('w', -1), ('x', 3), ('y', -1))),
                ('B', (('w', 2.0), ('x', 30.0), ('y', -1.0))),
                ('C', (('w', 3), ('x', -1), ('y', 2))),
                ('D', (('w', -1), ('x', -1), ('y', 3))),
            ),
        )

    # ---------------------------------------------------------------------------

    def test_frame_fillna_leading_a(self) -> None:
        a2 = np.array(
            [[None, None, None, None], [None, 1, None, 6], [None, 5, None, None]],
            dtype=object,
        )
        a1 = np.array([None, None, None], dtype=object)
        a3 = np.array([[None, 4], [None, 1], [None, 5]], dtype=object)
        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        f1 = Frame(
            tb1,
            index=self.get_letters(None, tb1.shape[0]),
            columns=self.get_letters(-tb1.shape[1], None),
        )

        self.assertEqual(
            f1.fillna_leading(0, axis=0).to_pairs(),
            (
                ('t', (('a', 0), ('b', 0), ('c', 0))),
                ('u', (('a', 0), ('b', 0), ('c', 0))),
                ('v', (('a', 0), ('b', 1), ('c', 5))),
                ('w', (('a', 0), ('b', 0), ('c', 0))),
                ('x', (('a', 0), ('b', 6), ('c', None))),
                ('y', (('a', 0), ('b', 0), ('c', 0))),
                ('z', (('a', 4), ('b', 1), ('c', 5))),
            ),
        )

        self.assertEqual(
            f1.fillna_leading(0, axis=1).to_pairs(),
            (
                ('t', (('a', 0), ('b', 0), ('c', 0))),
                ('u', (('a', 0), ('b', 0), ('c', 0))),
                ('v', (('a', 0), ('b', 1), ('c', 5))),
                ('w', (('a', 0), ('b', None), ('c', None))),
                ('x', (('a', 0), ('b', 6), ('c', None))),
                ('y', (('a', 0), ('b', None), ('c', None))),
                ('z', (('a', 4), ('b', 1), ('c', 5))),
            ),
        )

    def test_frame_fillfalsy_leading_a(self) -> None:
        a2 = np.array([['', '', '', ''], ['', 1, '', 6], ['', 5, '', '']], dtype=object)
        a1 = np.array(['', '', ''], dtype=object)
        a3 = np.array([['', 4], ['', 1], ['', 5]], dtype=object)
        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        f1 = Frame(
            tb1,
            index=self.get_letters(None, tb1.shape[0]),
            columns=self.get_letters(-tb1.shape[1], None),
        )

        self.assertEqual(
            f1.fillfalsy_leading(0, axis=0).to_pairs(),
            (
                ('t', (('a', 0), ('b', 0), ('c', 0))),
                ('u', (('a', 0), ('b', 0), ('c', 0))),
                ('v', (('a', 0), ('b', 1), ('c', 5))),
                ('w', (('a', 0), ('b', 0), ('c', 0))),
                ('x', (('a', 0), ('b', 6), ('c', ''))),
                ('y', (('a', 0), ('b', 0), ('c', 0))),
                ('z', (('a', 4), ('b', 1), ('c', 5))),
            ),
        )

        self.assertEqual(
            f1.fillfalsy_leading(0, axis=1).to_pairs(),
            (
                ('t', (('a', 0), ('b', 0), ('c', 0))),
                ('u', (('a', 0), ('b', 0), ('c', 0))),
                ('v', (('a', 0), ('b', 1), ('c', 5))),
                ('w', (('a', 0), ('b', ''), ('c', ''))),
                ('x', (('a', 0), ('b', 6), ('c', ''))),
                ('y', (('a', 0), ('b', ''), ('c', ''))),
                ('z', (('a', 4), ('b', 1), ('c', 5))),
            ),
        )

        with self.assertRaises(AxisInvalid):
            _ = f1.fillfalsy_leading(0, axis=-1)

    def test_frame_fillna_trailing_a(self) -> None:
        a2 = np.array(
            [[None, None, None, None], [None, 1, None, 6], [None, 5, None, None]],
            dtype=object,
        )
        a1 = np.array([None, None, None], dtype=object)
        a3 = np.array([[None, 4], [None, 1], [None, 5]], dtype=object)
        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        f1 = Frame(
            tb1,
            index=self.get_letters(None, tb1.shape[0]),
            columns=self.get_letters(-tb1.shape[1], None),
        )

        self.assertEqual(
            f1.fillna_trailing(0, axis=0).to_pairs(),
            (
                ('t', (('a', 0), ('b', 0), ('c', 0))),
                ('u', (('a', 0), ('b', 0), ('c', 0))),
                ('v', (('a', None), ('b', 1), ('c', 5))),
                ('w', (('a', 0), ('b', 0), ('c', 0))),
                ('x', (('a', None), ('b', 6), ('c', 0))),
                ('y', (('a', 0), ('b', 0), ('c', 0))),
                ('z', (('a', 4), ('b', 1), ('c', 5))),
            ),
        )

        self.assertEqual(
            f1.fillna_trailing(0, axis=1).to_pairs(),
            (
                ('t', (('a', None), ('b', None), ('c', None))),
                ('u', (('a', None), ('b', None), ('c', None))),
                ('v', (('a', None), ('b', 1), ('c', 5))),
                ('w', (('a', None), ('b', None), ('c', None))),
                ('x', (('a', None), ('b', 6), ('c', None))),
                ('y', (('a', None), ('b', None), ('c', None))),
                ('z', (('a', 4), ('b', 1), ('c', 5))),
            ),
        )

    def test_frame_fillfalsy_trailing_a(self) -> None:
        a2 = np.array([['', '', '', ''], ['', 1, '', 6], ['', 5, '', '']], dtype=object)
        a1 = np.array(['', '', ''], dtype=object)
        a3 = np.array([['', 4], ['', 1], ['', 5]], dtype=object)
        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        f1 = Frame(
            tb1,
            index=self.get_letters(None, tb1.shape[0]),
            columns=self.get_letters(-tb1.shape[1], None),
        )

        self.assertEqual(
            f1.fillfalsy_trailing(0, axis=0).to_pairs(),
            (
                ('t', (('a', 0), ('b', 0), ('c', 0))),
                ('u', (('a', 0), ('b', 0), ('c', 0))),
                ('v', (('a', ''), ('b', 1), ('c', 5))),
                ('w', (('a', 0), ('b', 0), ('c', 0))),
                ('x', (('a', ''), ('b', 6), ('c', 0))),
                ('y', (('a', 0), ('b', 0), ('c', 0))),
                ('z', (('a', 4), ('b', 1), ('c', 5))),
            ),
        )

        self.assertEqual(
            f1.fillfalsy_trailing(0, axis=1).to_pairs(),
            (
                ('t', (('a', ''), ('b', ''), ('c', ''))),
                ('u', (('a', ''), ('b', ''), ('c', ''))),
                ('v', (('a', ''), ('b', 1), ('c', 5))),
                ('w', (('a', ''), ('b', ''), ('c', ''))),
                ('x', (('a', ''), ('b', 6), ('c', ''))),
                ('y', (('a', ''), ('b', ''), ('c', ''))),
                ('z', (('a', 4), ('b', 1), ('c', 5))),
            ),
        )

        with self.assertRaises(AxisInvalid):
            _ = f1.fillfalsy_trailing(0, axis=-1)

    def test_frame_fillna_forward_a(self) -> None:
        a2 = np.array(
            [[8, None, None, None], [None, 1, None, 6], [0, 5, None, None]],
            dtype=object,
        )
        a1 = np.array([None, 3, None], dtype=object)
        a3 = np.array([[None, 4], [None, 1], [None, 5]], dtype=object)
        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        f1 = Frame(
            tb1,
            index=self.get_letters(None, tb1.shape[0]),
            columns=self.get_letters(-tb1.shape[1], None),
        )

        self.assertEqual(
            f1.fillna_forward().to_pairs(),
            (
                ('t', (('a', None), ('b', 3), ('c', 3))),
                ('u', (('a', 8), ('b', 8), ('c', 0))),
                ('v', (('a', None), ('b', 1), ('c', 5))),
                ('w', (('a', None), ('b', None), ('c', None))),
                ('x', (('a', None), ('b', 6), ('c', 6))),
                ('y', (('a', None), ('b', None), ('c', None))),
                ('z', (('a', 4), ('b', 1), ('c', 5))),
            ),
        )

        self.assertEqual(
            f1.fillna_backward().to_pairs(),
            (
                ('t', (('a', 3), ('b', 3), ('c', None))),
                ('u', (('a', 8), ('b', 0), ('c', 0))),
                ('v', (('a', 1), ('b', 1), ('c', 5))),
                ('w', (('a', None), ('b', None), ('c', None))),
                ('x', (('a', 6), ('b', 6), ('c', None))),
                ('y', (('a', None), ('b', None), ('c', None))),
                ('z', (('a', 4), ('b', 1), ('c', 5))),
            ),
        )

    def test_frame_fillna_forward_b(self) -> None:
        a2 = np.array(
            [[8, None, None, None], [None, 1, None, 6], [0, 5, None, None]],
            dtype=object,
        )
        a1 = np.array([None, 3, None], dtype=object)
        a3 = np.array([[None, 4], [None, 1], [None, 5]], dtype=object)
        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        f1 = Frame(
            tb1,
            index=self.get_letters(None, tb1.shape[0]),
            columns=self.get_letters(-tb1.shape[1], None),
        )
        # axis 1 tests
        self.assertEqual(
            f1.fillna_forward(axis=1).to_pairs(),
            (
                ('t', (('a', None), ('b', 3), ('c', None))),
                ('u', (('a', 8), ('b', 3), ('c', 0))),
                ('v', (('a', 8), ('b', 1), ('c', 5))),
                ('w', (('a', 8), ('b', 1), ('c', 5))),
                ('x', (('a', 8), ('b', 6), ('c', 5))),
                ('y', (('a', 8), ('b', 6), ('c', 5))),
                ('z', (('a', 4), ('b', 1), ('c', 5))),
            ),
        )

        self.assertEqual(
            f1.fillna_backward(axis=1).to_pairs(),
            (
                ('t', (('a', 8), ('b', 3), ('c', 0))),
                ('u', (('a', 8), ('b', 1), ('c', 0))),
                ('v', (('a', 4), ('b', 1), ('c', 5))),
                ('w', (('a', 4), ('b', 6), ('c', 5))),
                ('x', (('a', 4), ('b', 6), ('c', 5))),
                ('y', (('a', 4), ('b', 1), ('c', 5))),
                ('z', (('a', 4), ('b', 1), ('c', 5))),
            ),
        )

    def test_frame_fillna_forward_c(self) -> None:
        a2 = np.array(
            [[8, None, None, None], [None, 1, None, 6], [0, 5, None, None]],
            dtype=object,
        )
        a1 = np.array([None, 3, None], dtype=object)
        a3 = np.array([[None, 4], [None, None], [None, 5]], dtype=object)
        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        f1 = Frame(
            tb1,
            index=self.get_letters(None, tb1.shape[0]),
            columns=self.get_letters(-tb1.shape[1], None),
        )
        post = f1.fillna_forward(axis=1)

        self.assertEqual(
            f1.fillna_forward(1, axis=1).to_pairs(),
            (
                ('t', (('a', None), ('b', 3), ('c', None))),
                ('u', (('a', 8), ('b', 3), ('c', 0))),
                ('v', (('a', 8), ('b', 1), ('c', 5))),
                ('w', (('a', None), ('b', 1), ('c', 5))),
                ('x', (('a', None), ('b', 6), ('c', None))),
                ('y', (('a', None), ('b', 6), ('c', None))),
                ('z', (('a', 4), ('b', None), ('c', 5))),
            ),
        )

        self.assertEqual(
            f1.fillna_forward(2, axis=1).to_pairs(),
            (
                ('t', (('a', None), ('b', 3), ('c', None))),
                ('u', (('a', 8), ('b', 3), ('c', 0))),
                ('v', (('a', 8), ('b', 1), ('c', 5))),
                ('w', (('a', 8), ('b', 1), ('c', 5))),
                ('x', (('a', None), ('b', 6), ('c', 5))),
                ('y', (('a', None), ('b', 6), ('c', None))),
                ('z', (('a', 4), ('b', 6), ('c', 5))),
            ),
        )

        self.assertEqual(
            f1.fillna_backward(2, axis=1).to_pairs(),
            (
                ('t', (('a', 8), ('b', 3), ('c', 0))),
                ('u', (('a', 8), ('b', 1), ('c', 0))),
                ('v', (('a', None), ('b', 1), ('c', 5))),
                ('w', (('a', None), ('b', 6), ('c', None))),
                ('x', (('a', 4), ('b', 6), ('c', 5))),
                ('y', (('a', 4), ('b', None), ('c', 5))),
                ('z', (('a', 4), ('b', None), ('c', 5))),
            ),
        )

    def test_frame_fillfalsy_forward_a(self) -> None:
        a2 = np.array([[8, '', '', ''], ['', 1, '', 6], [1, 5, '', '']], dtype=object)
        a1 = np.array(['', 3, ''], dtype=object)
        a3 = np.array([['', 4], ['', 1], ['', 5]], dtype=object)
        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        f1 = Frame(
            tb1,
            index=self.get_letters(None, tb1.shape[0]),
            columns=self.get_letters(-tb1.shape[1], None),
        )

        self.assertEqual(
            f1.fillfalsy_forward().to_pairs(),
            (
                ('t', (('a', ''), ('b', 3), ('c', 3))),
                ('u', (('a', 8), ('b', 8), ('c', 1))),
                ('v', (('a', ''), ('b', 1), ('c', 5))),
                ('w', (('a', ''), ('b', ''), ('c', ''))),
                ('x', (('a', ''), ('b', 6), ('c', 6))),
                ('y', (('a', ''), ('b', ''), ('c', ''))),
                ('z', (('a', 4), ('b', 1), ('c', 5))),
            ),
        )

        with self.assertRaises(AxisInvalid):
            f1.fillfalsy_forward(axis=-1)

        self.assertEqual(
            f1.fillfalsy_backward().to_pairs(),
            (
                ('t', (('a', 3), ('b', 3), ('c', ''))),
                ('u', (('a', 8), ('b', 1), ('c', 1))),
                ('v', (('a', 1), ('b', 1), ('c', 5))),
                ('w', (('a', ''), ('b', ''), ('c', ''))),
                ('x', (('a', 6), ('b', 6), ('c', ''))),
                ('y', (('a', ''), ('b', ''), ('c', ''))),
                ('z', (('a', 4), ('b', 1), ('c', 5))),
            ),
        )

        with self.assertRaises(AxisInvalid):
            f1.fillfalsy_backward(axis=-1)

    def test_frame_fillfalsy_forward_b(self) -> None:
        a2 = np.array([[8, '', '', ''], ['', 1, '', 6], [1, 5, '', '']], dtype=object)
        a1 = np.array(['', 3, ''], dtype=object)
        a3 = np.array([['', 4], ['', 1], ['', 5]], dtype=object)
        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        f1 = Frame(
            tb1,
            index=self.get_letters(None, tb1.shape[0]),
            columns=self.get_letters(-tb1.shape[1], None),
        )
        self.assertEqual(
            f1.fillfalsy_forward(axis=1).to_pairs(),
            (
                ('t', (('a', ''), ('b', 3), ('c', ''))),
                ('u', (('a', 8), ('b', 3), ('c', 1))),
                ('v', (('a', 8), ('b', 1), ('c', 5))),
                ('w', (('a', 8), ('b', 1), ('c', 5))),
                ('x', (('a', 8), ('b', 6), ('c', 5))),
                ('y', (('a', 8), ('b', 6), ('c', 5))),
                ('z', (('a', 4), ('b', 1), ('c', 5))),
            ),
        )

        self.assertEqual(
            f1.fillfalsy_backward(2, axis=1).to_pairs(),
            (
                ('t', (('a', 8), ('b', 3), ('c', 1))),
                ('u', (('a', 8), ('b', 1), ('c', 1))),
                ('v', (('a', ''), ('b', 1), ('c', 5))),
                ('w', (('a', ''), ('b', 6), ('c', ''))),
                ('x', (('a', 4), ('b', 6), ('c', 5))),
                ('y', (('a', 4), ('b', 1), ('c', 5))),
                ('z', (('a', 4), ('b', 1), ('c', 5))),
            ),
        )

    def test_frame_empty_a(self) -> None:
        f1 = FrameGO(index=('a', 'b', 'c'))
        f1['w'] = Series.from_items(zip('cebga', (10, 20, 30, 40, 50)))
        f1['x'] = Series.from_items(zip('abc', range(3, 6)))
        f1['y'] = Series.from_items(zip('abcd', range(2, 6)))
        f1['z'] = Series.from_items(zip('qabc', range(7, 11)))

        self.assertEqual(
            f1.to_pairs(),
            (
                ('w', (('a', 50), ('b', 30), ('c', 10))),
                ('x', (('a', 3), ('b', 4), ('c', 5))),
                ('y', (('a', 2), ('b', 3), ('c', 4))),
                ('z', (('a', 8), ('b', 9), ('c', 10))),
            ),
        )

    # ---------------------------------------------------------------------------
    @skip_win
    def test_frame_from_csv_a(self) -> None:
        # header, mixed types, no index

        s1 = StringIO(
            'count,score,color\n1,1.3,red\n3,5.2,green\n100,3.4,blue\n4,9.0,black'
        )
        f1 = Frame.from_csv(s1)

        post = f1.iloc[:, :2].sum(axis=0)
        self.assertEqual(post.to_pairs(), (('count', 108.0), ('score', 18.9)))
        self.assertEqual(f1.shape, (4, 3))

        self.assertEqual(
            f1.dtypes.iter_element().apply(str).to_pairs(),
            (('count', 'int64'), ('score', 'float64'), ('color', '<U5')),
        )

        s2 = StringIO(
            'color,count,score\nred,1,1.3\ngreen,3,5.2\nblue,100,3.4\nblack,4,9.0'
        )

        f2 = Frame.from_csv(s2)
        self.assertEqual(
            f2['count':].sum().to_pairs(),  # type: ignore  # https://github.com/python/typeshed/pull/3024
            (('count', 108.0), ('score', 18.9)),
        )
        self.assertEqual(f2.shape, (4, 3))
        self.assertEqual(
            f2.dtypes.iter_element().apply(str).to_pairs(),
            (('color', '<U5'), ('count', 'int64'), ('score', 'float64')),
        )

        # add junk at beginning and end
        s3 = StringIO(
            'junk\ncolor,count,score\nred,1,1.3\ngreen,3,5.2\nblue,100,3.4\nblack,4,9.0\njunk'
        )

        f3 = Frame.from_csv(s3, skip_header=1, skip_footer=1)
        self.assertEqual(f3.shape, (4, 3))
        self.assertEqual(
            f3.dtypes.iter_element().apply(str).to_pairs(),
            (('color', '<U5'), ('count', 'int64'), ('score', 'float64')),
        )

    def test_frame_from_csv_b(self) -> None:
        filelike = StringIO("""count,number,weight,scalar,color,active
0,4,234.5,5.3,'red',False
30,50,9.234,5.434,'blue',True""")
        f1 = Frame.from_csv(filelike)

        self.assertEqual(
            f1.columns.values.tolist(),
            ['count', 'number', 'weight', 'scalar', 'color', 'active'],
        )

    def test_frame_from_csv_c(self) -> None:
        s1 = StringIO(
            'color,count,score\nred,1,1.3\ngreen,3,5.2\nblue,100,3.4\nblack,4,9.0'
        )
        f1 = Frame.from_csv(s1, index_depth=1)
        self.assertEqual(
            f1.to_pairs(),
            (
                ('count', (('red', 1), ('green', 3), ('blue', 100), ('black', 4))),
                (
                    'score',
                    (('red', 1.3), ('green', 5.2), ('blue', 3.4), ('black', 9.0)),
                ),
            ),
        )

    def test_frame_from_csv_d(self) -> None:
        s1 = StringIO('color,count,score\n')
        f1 = Frame.from_csv(s1, columns_depth=1)
        self.assertEqual(f1.to_pairs(), (('color', ()), ('count', ()), ('score', ())))

    def test_frame_from_csv_e(self) -> None:
        s1 = StringIO(
            'group,count,score,color\nA,1,1.3,red\nA,3,5.2,green\nB,100,3.4,blue\nB,4,9.0,black'
        )

        f1 = sf.Frame.from_csv(s1, index_depth=2, columns_depth=1)
        self.assertEqual(f1.index.__class__, IndexHierarchy)
        self.assertEqual(
            f1.to_pairs(),
            (
                (
                    'score',
                    (
                        (('A', 1), 1.3),
                        (('A', 3), 5.2),
                        (('B', 100), 3.4),
                        (('B', 4), 9.0),
                    ),
                ),
                (
                    'color',
                    (
                        (('A', 1), 'red'),
                        (('A', 3), 'green'),
                        (('B', 100), 'blue'),
                        (('B', 4), 'black'),
                    ),
                ),
            ),
        )

    def test_frame_from_csv_f(self) -> None:
        s1 = StringIO(
            'group,count,score,color\nA,nan,1.3,red\nB,NaN,5.2,green\nC,NAN,3.4,blue\nD,,9.0,black'
        )

        f1 = sf.Frame.from_csv(s1, index_depth=1, columns_depth=1)

        self.assertAlmostEqualFramePairs(
            f1.to_pairs(),
            (
                ('count', (('A', np.nan), ('B', np.nan), ('C', np.nan), ('D', np.nan))),
                ('score', (('A', 1.3), ('B', 5.2), ('C', 3.4), ('D', 9.0))),
                (
                    'color',
                    (('A', 'red'), ('B', 'green'), ('C', 'blue'), ('D', 'black')),
                ),
            ),
        )

    def test_frame_from_csv_g(self) -> None:
        filelike = StringIO("""0,4,234.5,5.3,'red',False
30,50,9.234,5.434,'blue',True""")
        f1 = Frame.from_csv(filelike, columns_depth=0)
        self.assertEqual(
            f1.to_pairs(),
            (
                (0, ((0, 0), (1, 30))),
                (1, ((0, 4), (1, 50))),
                (2, ((0, 234.5), (1, 9.234))),
                (3, ((0, 5.3), (1, 5.434))),
                (4, ((0, "'red'"), (1, "'blue'"))),
                (5, ((0, False), (1, True))),
            ),
        )

    def test_frame_from_csv_h1(self) -> None:
        s1 = StringIO(
            'group,count,score,color\nA,nan,1.3,red\nB,NaN,5.2,green\nC,NULL,3.4,blue\nD,,9.0,black'
        )

        f1 = sf.Frame.from_csv(
            s1, index_depth=1, columns_depth=1, dtypes=dict(score=np.float16)
        )

        self.assertEqual(
            f1.dtypes.to_pairs(),
            (
                ('count', np.dtype('<U4')),
                ('score', np.dtype('float16')),
                ('color', np.dtype('<U5')),
            ),
        )

    def test_frame_from_csv_h2(self) -> None:
        s1 = StringIO(
            'group,count,score,color\nA,nan,1.3,red\nB,NaN,5.2,green\nC,NULL,3.4,blue\nD,,9.0,black'
        )

        f1 = sf.Frame.from_csv(
            s1,
            index_depth=1,
            columns_depth=1,
            store_filter=StoreFilter(),
            dtypes=(str, str, np.float16, str),
        )
        self.assertEqual(
            f1.dtypes.to_pairs(),
            (
                ('count', np.dtype('O')),
                ('score', np.dtype('float16')),
                ('color', np.dtype('<U5')),
            ),
        )
        self.assertEqual(
            f1['count'].astype(str).values.tolist(), ['nan', 'nan', 'nan', 'nan']
        )

    def test_frame_from_csv_i(self) -> None:
        s1 = StringIO('1,2,3\n4,5,6')

        f1 = sf.Frame.from_csv(
            s1, index_depth=0, columns_depth=0, dtypes=[np.int64, str, np.int64]
        )
        self.assertEqual(
            f1.dtypes.values.tolist(),
            [np.dtype('int64'), np.dtype('<U1'), np.dtype('int64')],
        )

    def test_frame_from_csv_j(self) -> None:
        s1 = StringIO('1,2,3\n4,5,6')

        f2 = sf.Frame.from_csv(
            s1, index_depth=2, columns_depth=0, dtypes=[np.int64, str, np.int64]
        )

        self.assertEqual(f2.to_pairs(axis=0), ((0, (((1, '2'), 3), ((4, '5'), 6))),))

    def test_frame_from_csv_k(self) -> None:
        s1 = StringIO('1,2,3,4\n')
        f1 = Frame.from_csv(s1, index_depth=0, columns_depth=0)
        self.assertEqual(
            f1.to_pairs(),
            ((0, ((0, 1),)), (1, ((0, 2),)), (2, ((0, 3),)), (3, ((0, 4),))),
        )

    def test_frame_from_csv_m(self) -> None:
        s1 = StringIO('a,-,b,-\nx,y,x,y\n1,2,3,10\n4,5,6,20')

        f2 = sf.Frame.from_csv(
            s1,
            index_depth=0,
            columns_depth=2,
            dtypes=int,
            columns_continuation_token='-',
        )
        self.assertEqual(
            f2.to_pairs(axis=0),
            (
                (('a', 'x'), ((0, 1), (1, 4))),
                (('a', 'y'), ((0, 2), (1, 5))),
                (('b', 'x'), ((0, 3), (1, 6))),
                (('b', 'y'), ((0, 10), (1, 20))),
            ),
        )

    def test_frame_from_csv_n(self) -> None:
        s1 = StringIO(',1,a,b\nX,1,43,54\n-,2,1,3\nY,1,8,10\n-,2,6,20')

        f2 = sf.Frame.from_csv(
            s1,
            index_depth=2,
            columns_depth=1,
            # dtypes=int,.
            index_continuation_token='-',
        )

        self.assertEqual(
            f2.to_pairs(),
            (
                ('a', ((('X', 1), 43), (('X', 2), 1), (('Y', 1), 8), (('Y', 2), 6))),
                ('b', ((('X', 1), 54), (('X', 2), 3), (('Y', 1), 10), (('Y', 2), 20))),
            ),
        )

    def test_frame_from_csv_o(self) -> None:
        s1 = StringIO(',1,a,b\n-,1,43,54\nX,2,1,3\nY,1,8,10\n-,2,6,20')

        f2 = sf.Frame.from_csv(
            s1,
            index_depth=2,
            columns_depth=1,
            # dtypes=int,.
            index_continuation_token='-',
        )

        self.assertEqual(
            f2.to_pairs(),
            (
                ('a', ((('-', 1), 43), (('X', 2), 1), (('Y', 1), 8), (('Y', 2), 6))),
                ('b', ((('-', 1), 54), (('X', 2), 3), (('Y', 1), 10), (('Y', 2), 20))),
            ),
        )

    def test_frame_from_csv_p(self) -> None:
        s1 = ['junk', ',x,y', 'a,3,2', 'b,5,1', 'junk', 'junk']
        f1 = sf.Frame.from_csv(
            s1, index_depth=1, columns_depth=1, skip_footer=2, skip_header=1
        )
        self.assertEqual(
            f1.to_pairs(), (('x', (('a', 3), ('b', 5))), ('y', (('a', 2), ('b', 1))))
        )

    def test_frame_from_csv_q(self) -> None:
        s1 = StringIO('x,y,x,y\n1,2,3,10\n4,5,6,20\njunk')

        f1 = sf.Frame.from_csv(
            s1,
            index_depth=0,
            columns_depth=0,
            skip_footer=1,
            skip_header=1,
        )
        self.assertEqual(
            f1.to_pairs(),
            (
                (0, ((0, 1), (1, 4))),
                (1, ((0, 2), (1, 5))),
                (2, ((0, 3), (1, 6))),
                (3, ((0, 10), (1, 20))),
            ),
        )

    def test_frame_from_csv_r1(self) -> None:
        s1 = StringIO('1,2,3,4\n4,5,6,5')

        f1 = sf.Frame.from_csv(
            s1,
            index_depth=1,
            index_column_first=2,
            columns_depth=0,
        )
        self.assertEqual(
            f1.to_pairs(),
            ((0, ((3, 1), (6, 4))), (1, ((3, 2), (6, 5))), (2, ((3, 4), (6, 5)))),
        )

    def test_frame_from_csv_r2(self) -> None:
        s1 = StringIO('1,2,3,4\n4,5,6,5')
        with self.assertRaises(ErrorInitFrame):
            f1 = sf.Frame.from_csv(
                s1,
                index_depth=1,
                index_column_first=2,
                columns_depth=1,
            )

    def test_frame_from_csv_r3(self) -> None:
        s1 = StringIO('1,2,3,4\n4,5,6,5')
        with self.assertRaises(ErrorInitFrame):
            f1 = sf.Frame.from_csv(
                s1,
                index_depth=1,
                index_column_first='foo',
                columns_depth=0,
            )

    def test_frame_from_csv_s(self) -> None:
        s1 = StringIO('A,A,B,B\nNone,x,None,y\n1,2,3,4\n4,5,6,5\n')

        f1 = sf.Frame.from_csv(
            s1, index_depth=0, columns_depth=2, store_filter=StoreFilter()
        )
        self.assertEqual(f1.columns.depth, 2)
        self.assertEqual(
            f1.columns.values.tolist(),
            [['A', None], ['A', 'x'], ['B', None], ['B', 'y']],
        )
        self.assertEqual(f1.shape, (2, 4))

    def test_frame_from_csv_t(self) -> None:
        s1 = StringIO('A,-,B\nNone,x,None,y\n1,2,3,4\n4,5,6,5\n')

        f1 = sf.Frame.from_csv(
            s1,
            index_depth=0,
            columns_depth=2,
            store_filter=StoreFilter(),
            columns_continuation_token='-',
        )
        self.assertEqual(f1.columns.depth, 2)
        self.assertEqual(
            f1.columns.values.tolist(),
            [['A', None], ['A', 'x'], ['B', None], ['B', 'y']],
        )
        self.assertEqual(f1.shape, (2, 4))

    def test_frame_from_csv_u(self) -> None:
        s1 = StringIO('1,2,3,4\n4,5,6,5')
        f1 = sf.Frame.from_csv(
            s1,
            index_depth=0,
            consolidate_blocks=True,
            columns_depth=0,
        )
        self.assertTrue(f1._blocks.unified)

    # ---------------------------------------------------------------------------

    @skip_win
    def test_structured_array_to_d_ia_cl_a(self) -> None:
        a1 = np.array(np.arange(12).reshape((3, 4)))
        post, _, _ = Frame._structured_array_to_d_ia_cl(
            a1, dtypes=[np.int64, str, np.int64, str]
        )

        self.assertEqual(
            post.dtypes.tolist(),
            [np.dtype('int64'), np.dtype('<U21'), np.dtype('int64'), np.dtype('<U21')],
        )

    def test_structured_arrayto_d_ia_cl_b(self) -> None:
        a1 = np.array(np.arange(12).reshape((3, 4)))
        post, _, _ = Frame._structured_array_to_d_ia_cl(
            a1,
            dtypes=[np.int64, str, str, str],
            consolidate_blocks=True,
        )
        self.assertEqual(post.shapes.tolist(), [(3,), (3, 3)])

    def test_structured_arrayto_d_ia_cl_c(self) -> None:
        a1 = np.array(np.arange(12).reshape((3, 4)))

        with self.assertRaises(ErrorInitFrame):
            # cannot specify index_column_first if index_depth is 0
            post, _, _ = Frame._structured_array_to_d_ia_cl(
                a1,
                index_depth=0,
                index_column_first=1,
                dtypes=[np.int64, str, str, str],
                consolidate_blocks=True,
            )

    def test_structured_array_to_d_ia_cl_d(self) -> None:
        a1 = np.array(
            [(3,)],
            dtype=[
                ('a', int),
            ],
        )
        post, _, _ = Frame._structured_array_to_d_ia_cl(
            a1,
        )
        self.assertEqual(post.shape, (1, 1))

    # ---------------------------------------------------------------------------

    def test_from_data_index_arrays_column_labels_a(self) -> None:
        tb = TypeBlocks.from_blocks(np.array([3, 4, 5]))

        f1 = Frame._from_data_index_arrays_column_labels(
            tb,
            index_depth=0,
            index_arrays=(),
            index_constructors=None,
            columns_depth=0,
            columns_labels=(),
            columns_constructors=None,
            name='foo',
        )
        self.assertEqual(f1.to_pairs(), ((0, ((0, 3), (1, 4), (2, 5))),))

    # ---------------------------------------------------------------------------

    def test_frame_from_delimited_a(self) -> None:
        with temp_file('.txt', path=True) as fp:
            with open(fp, 'w', encoding='utf-8') as file:
                file.write('\n'.join(('index|A|B', 'a|True|20.2', 'b|False|85.3')))
                file.close()

            with self.assertRaises(ErrorInitFrame):
                f = Frame.from_delimited(fp, index_depth=1, delimiter='|', skip_header=-1)

            f = Frame.from_delimited(fp, index_depth=1, delimiter='|')
            self.assertEqual(
                f.to_pairs(),
                (('A', (('a', True), ('b', False))), ('B', (('a', 20.2), ('b', 85.3)))),
            )

    def test_frame_from_delimited_b(self) -> None:
        with temp_file('.txt', path=True) as fp:
            with open(fp, 'w', encoding='utf-8') as file:
                file.write(
                    '\n'.join(('index|A|B', 'false|false|true', 'true|true|false'))
                )
                file.close()

            # dtypes are applied to all columns, even those that will become index
            f1 = Frame.from_delimited(
                fp,
                index_depth=1,
                columns_depth=1,
                delimiter='|',
                dtypes=bool,
            )

            self.assertEqual(
                f1.to_pairs(),
                (
                    ('A', ((False, False), (True, True))),
                    ('B', ((False, True), (True, False))),
                ),
            )

    def test_frame_from_delimited_c(self) -> None:
        dtypes = (int, str, int, int)
        msg = 'a|b|c|d\n1940|2021-04-03|3|5\n1492|1743-04-03|-4|9\n'
        self.assertEqual(
            Frame.from_delimited(
                msg.split('\n'), delimiter='|', dtypes=dtypes
            ).to_pairs(),
            (
                ('a', ((0, 1940), (1, 1492))),
                ('b', ((0, '2021-04-03'), (1, '1743-04-03'))),
                ('c', ((0, 3), (1, -4))),
                ('d', ((0, 5), (1, 9))),
            ),
        )

        f1 = Frame.from_delimited(
            msg.split('\n'),
            delimiter='|',
            dtypes=dtypes,
            index_constructors=IndexYear,
            index_depth=1,
        )
        self.assertEqual(f1.index.dtype, np.dtype('<M8[Y]'))  # type: ignore

        f2 = Frame.from_delimited(
            msg.split('\n'),
            delimiter='|',
            dtypes=dtypes,
            index_constructors=(IndexYear,),
            index_depth=1,
        )
        self.assertEqual(f2.index.dtype, np.dtype('<M8[Y]'))  # type: ignore

        with self.assertRaises(RuntimeError):
            _ = Frame.from_delimited(
                msg.split('\n'),
                delimiter='|',
                dtypes=dtypes,
                index_constructors=(IndexYear, IndexDate),
                index_depth=1,
            )

        f3 = Frame.from_delimited(
            msg.split('\n'),
            delimiter='|',
            index_constructors=(IndexYear, IndexDate),
            index_depth=2,
            dtypes=dtypes,
        )
        self.assertEqual(f3.index.depth, 2)
        self.assertEqual(f3.index.index_types.values.tolist(), [IndexYear, IndexDate])

    def test_frame_from_delimited_d(self) -> None:
        msg = '1930|1931\n2021-01-01|2022-04-03\n3|5\n-4|9\n'

        f1 = Frame.from_delimited(
            msg.split('\n'),
            delimiter='|',
            columns_constructors=IndexYear,
            columns_depth=1,
        )
        self.assertEqual(f1.columns.__class__, IndexYear)

        with self.assertRaises(RuntimeError):
            _ = Frame.from_delimited(
                msg.split('\n'),
                delimiter='|',
                columns_constructors=(IndexYear, IndexDate),
                columns_depth=1,
            )

        f2 = Frame.from_delimited(
            msg.split('\n'),
            delimiter='|',
            columns_constructors=IndexYear,
            columns_depth=2,
        )
        self.assertEqual(f2.columns.index_types.values.tolist(), [IndexYear, IndexYear])

        f3 = Frame.from_delimited(
            msg.split('\n'),
            delimiter='|',
            columns_constructors=(IndexYear, IndexDate),
            columns_depth=2,
        )
        self.assertEqual(f3.columns.index_types.values.tolist(), [IndexYear, IndexDate])

    def test_frame_from_delimited_e(self) -> None:
        msg = 'a|a|b|b\n0|1|0|1\nw|x|y|z\n'

        f1 = Frame.from_delimited(
            msg.split('\n'),
            delimiter='|',
            index_depth=0,
            columns_depth=2,
        )
        self.assertEqual(
            f1.to_pairs(),
            (
                (('a', 0), ((0, 'w'),)),
                (('a', 1), ((0, 'x'),)),
                (('b', 0), ((0, 'y'),)),
                (('b', 1), ((0, 'z'),)),
            ),
        )

    def test_frame_from_delimited_f(self) -> None:
        msg = 'a|b|c\n0|1|FALSE\n23|1|true\n'

        f1 = Frame.from_delimited(
            msg.split('\n'),
            delimiter='|',
            index_depth=0,
            columns_depth=1,
            dtypes={'c': bool, 'd': float},  # NOTE: can overspecify
        )
        self.assertEqual([d.kind for d in f1.dtypes.values], ['i', 'i', 'b'])
        self.assertEqual(
            f1.to_pairs(),
            (
                ('a', ((0, 0), (1, 23))),
                ('b', ((0, 1), (1, 1))),
                ('c', ((0, False), (1, True))),
            ),
        )

    def test_frame_from_delimited_g(self) -> None:
        with temp_file('.txt', path=True) as fp:
            with open(fp, 'w', encoding='utf-8') as file:
                file.write(
                    '\n'.join(
                        ('index|A|B', 'false|false|true', 'true|true|false', 'junk')
                    )
                )
                file.close()

            # dtypes are applied to all columns, even those that will become index
            f1 = Frame.from_delimited(
                fp,
                index_depth=0,
                columns_depth=0,
                delimiter='|',
                dtypes=bool,
                skip_header=1,
                skip_footer=1,
            )
            self.assertEqual(
                f1.to_pairs(),
                (
                    (0, ((0, False), (1, True))),
                    (1, ((0, False), (1, True))),
                    (2, ((0, True), (1, False))),
                ),
            )

    def test_frame_from_delimited_h(self) -> None:
        msg = 'a|b|c|d\nFalse|1|0|q\nTrue|4|5|z\nFalse|9|6|w'

        with self.assertRaises(ErrorInitFrame):
            f1 = Frame.from_delimited(
                msg.split('\n'),
                delimiter='|',
                index_depth=0,
                columns_depth=1,
                index_column_first='c',
            )

    def test_frame_from_delimited_i(self) -> None:
        msg = 'False|1|0|q\nTrue|4|5|z\nFalse|9|6|w'

        f1 = Frame.from_delimited(
            msg.split('\n'),
            delimiter='|',
            index_depth=1,
            columns_depth=0,
            index_column_first=2,
        )
        self.assertEqual(
            f1.to_pairs(),
            (
                (0, ((0, False), (5, True), (6, False))),
                (1, ((0, 1), (5, 4), (6, 9))),
                (2, ((0, 'q'), (5, 'z'), (6, 'w'))),
            ),
        )

    def test_frame_from_delimited_j(self) -> None:
        msg = 'a|b|c|d\nFalse|1|0|q\nTrue|4|5|z\nFalse|9|6|w'

        f1 = Frame.from_delimited(
            msg.split('\n'),
            delimiter='|',
            index_depth=0,
            columns_depth=1,
            columns_select=('a', 'd'),
        )
        self.assertEqual(
            f1.to_pairs(),
            (
                ('a', ((0, False), (1, True), (2, False))),
                ('d', ((0, 'q'), (1, 'z'), (2, 'w'))),
            ),
        )

    def test_frame_from_delimited_k(self) -> None:
        msg = 'a|b|c|d\nFalse|1|0|q\nTrue|4|5|z\nFalse|9|6|w'

        f1 = Frame.from_delimited(
            msg.split('\n'),
            delimiter='|',
            index_depth=0,
            columns_depth=1,
            columns_select=('d',),
        )
        self.assertEqual(f1.to_pairs(), (('d', ((0, 'q'), (1, 'z'), (2, 'w'))),))

    def test_frame_from_delimited_l(self) -> None:
        msg = 'False|1|0|q\nTrue|4|5|z\nFalse|9|6|w'

        f1 = Frame.from_delimited(
            msg.split('\n'),
            delimiter='|',
            index_depth=0,
            columns_depth=0,
            columns_select=(1, 2),
        )
        self.assertEqual(
            f1.to_pairs(),
            ((0, ((0, 1), (1, 4), (2, 9))), (1, ((0, 0), (1, 5), (2, 6)))),
        )

    def test_frame_from_delimited_m(self) -> None:
        msg = 'False|1|0|q\nTrue|4|5|z\nFalse|9|6|w'

        with self.assertRaises(ErrorInitFrame):
            f1 = Frame.from_delimited(
                msg.split('\n'),
                delimiter='|',
                index_depth=1,
                columns_depth=0,
                columns_select=(1, 2),
            )

    def test_frame_from_delimited_n(self) -> None:
        msg = '2021-01-02|1|0|q\n2022-03-05|4|5|z\n1983-08-08|9|6|w'

        f1 = Frame.from_delimited(
            msg.split('\n'),
            delimiter='|',
            index_depth=1,
            columns_depth=0,
            index_constructors=IndexDate,
        )
        self.assertIs(f1.index.__class__, IndexDate)
        dt64 = np.datetime64
        self.assertEqual(
            f1.to_pairs(),
            (
                (
                    0,
                    (
                        (dt64('2021-01-02'), 1),
                        (dt64('2022-03-05'), 4),
                        (dt64('1983-08-08'), 9),
                    ),
                ),
                (
                    1,
                    (
                        (dt64('2021-01-02'), 0),
                        (dt64('2022-03-05'), 5),
                        (dt64('1983-08-08'), 6),
                    ),
                ),
                (
                    2,
                    (
                        (dt64('2021-01-02'), 'q'),
                        (dt64('2022-03-05'), 'z'),
                        (dt64('1983-08-08'), 'w'),
                    ),
                ),
            ),
        )

    def test_frame_from_delimited_o(self) -> None:
        msg = '2021-01-02|a|0|q\n2022-01-02|b|5|z\n1983-08-08|a|6|w'

        f1 = Frame.from_delimited(
            msg.split('\n'),
            delimiter='|',
            index_depth=2,
            columns_depth=0,
            index_constructors=(IndexDate, Index),
        )
        self.assertEqual(f1.index.index_types.values.tolist(), [IndexDate, Index])
        dt64 = np.datetime64
        self.assertEqual(
            f1.to_pairs(),
            (
                (
                    0,
                    (
                        ((dt64('2021-01-02'), 'a'), 0),
                        ((dt64('2022-01-02'), 'b'), 5),
                        ((dt64('1983-08-08'), 'a'), 6),
                    ),
                ),
                (
                    1,
                    (
                        ((dt64('2021-01-02'), 'a'), 'q'),
                        ((dt64('2022-01-02'), 'b'), 'z'),
                        ((dt64('1983-08-08'), 'a'), 'w'),
                    ),
                ),
            ),
        )

    def test_frame_from_delimited_p(self) -> None:
        msg = 'a\\||1|0|q\nb\\||4|5|z\nc\\||9|6|w'

        f1 = Frame.from_delimited(
            msg.split('\n'),
            delimiter='|',
            index_depth=1,
            columns_depth=0,
            escape_char='\\',
        )
        self.assertEqual(f1.shape, (3, 3))
        self.assertEqual(f1.index.values.tolist(), ['a|', 'b|', 'c|'])
        self.assertEqual(
            f1.to_pairs(),
            (
                (0, (('a|', 1), ('b|', 4), ('c|', 9))),
                (1, (('a|', 0), ('b|', 5), ('c|', 6))),
                (2, (('a|', 'q'), ('b|', 'z'), ('c|', 'w'))),
            ),
        )

    def test_frame_from_delimited_q(self) -> None:
        msg = "'a|'|1|0|q\n'b|'|4|5|z\n'c|'|9|6|w"

        f1 = Frame.from_delimited(
            msg.split('\n'),
            delimiter='|',
            index_depth=1,
            columns_depth=0,
            quote_char="'",
        )
        self.assertEqual(
            f1.to_pairs(),
            (
                (0, (('a|', 1), ('b|', 4), ('c|', 9))),
                (1, (('a|', 0), ('b|', 5), ('c|', 6))),
                (2, (('a|', 'q'), ('b|', 'z'), ('c|', 'w'))),
            ),
        )

    def test_frame_from_delimited_r(self) -> None:
        msg = '0|1|3\n4|0|6\n'
        f1 = Frame.from_delimited(
            msg.split('\n'),
            delimiter='|',
            index_depth=0,
            columns_depth=0,
            dtypes={1: bool, 2: str},
        )
        self.assertEqual([dt.kind for dt in f1.dtypes.values], ['i', 'b', 'U'])

    def test_frame_from_delimited_s(self) -> None:
        msg = '0|1|3\n4|0|6\n'
        f1 = Frame.from_delimited(
            msg.split('\n'),
            delimiter='|',
            index_depth=0,
            columns_depth=0,
            dtypes=chain((int,), repeat(str)),
        )
        self.assertEqual([dt.kind for dt in f1.dtypes.values], ['i', 'U', 'U'])

    def test_frame_from_delimited_t(self) -> None:
        msg = '0|1|3\n4|0|6\n'
        f1 = Frame.from_delimited(
            msg.split('\n'),
            delimiter='|',
            index_depth=0,
            columns_depth=0,
            dtypes=defaultdict(lambda: str, {1: int}),  # type: ignore
        )
        self.assertEqual([dt.kind for dt in f1.dtypes.values], ['U', 'i', 'U'])

    def test_frame_from_delimited_u(self) -> None:
        dtypes = ('datetime64[D]', 'datetime64[D]', str, str)
        msg = 'date_close,balance_open,balance_close\n2017-06-30,0.00,193.33\n2017-07-31,193.33,197.97\n2017-08-31,197.97,203.02\n2017-09-30,203.02,206.44'
        with self.assertRaises(ValueError):
            f = Frame.from_delimited(msg.split('\n'), delimiter=',', dtypes=dtypes)

    # ---------------------------------------------------------------------------

    def test_frame_from_tsv_a(self) -> None:
        with temp_file('.txt', path=True) as fp:
            with open(fp, 'w', encoding='utf-8') as file:
                file.write('\n'.join(('index\tA\tB', 'a\tTrue\t20.2', 'b\tFalse\t85.3')))
                file.close()

            f = Frame.from_tsv(fp, index_depth=1, dtypes={'A': bool})
            self.assertEqual(
                f.to_pairs(),
                (('A', (('a', True), ('b', False))), ('B', (('a', 20.2), ('b', 85.3)))),
            )

    def test_frame_from_tsv_b(self) -> None:
        # a generator of delimited strings also works

        def lines() -> tp.Iterator[str]:
            yield 'a\tb\tc\td'
            for i in range(4):
                yield f'{i}\t{i + 1}\t{i + 2}\t{i + 3}'

        f = Frame.from_tsv(lines())
        self.assertEqual(
            f.to_pairs(),
            (
                ('a', ((0, 0), (1, 1), (2, 2), (3, 3))),
                ('b', ((0, 1), (1, 2), (2, 3), (3, 4))),
                ('c', ((0, 2), (1, 3), (2, 4), (3, 5))),
                ('d', ((0, 3), (1, 4), (2, 5), (3, 6))),
            ),
        )

    def test_frame_from_tsv_c(self) -> None:
        input_stream = StringIO("""
        196412	0.0
        196501	0.0
        196502	0.0
        196503	0.0
        196504	0.0
        196505	0.0""")

        f1 = sf.Frame.from_tsv(input_stream, index_depth=1, columns_depth=0)

        self.assertEqual(
            f1.to_pairs(),
            (
                (
                    0,
                    (
                        (196412, 0.0),
                        (196501, 0.0),
                        (196502, 0.0),
                        (196503, 0.0),
                        (196504, 0.0),
                        (196505, 0.0),
                    ),
                ),
            ),
        )

        input_stream = StringIO("""
        196412	0.0	0.1
        196501	0.0	0.1
        196502	0.0	0.1
        196503	0.0	0.1
        196504	0.0	0.1
        196505	0.0	0.1""")

        f2 = sf.Frame.from_tsv(input_stream, index_depth=1, columns_depth=0)

        self.assertEqual(
            f2.to_pairs(),
            (
                (
                    0,
                    (
                        (196412, 0.0),
                        (196501, 0.0),
                        (196502, 0.0),
                        (196503, 0.0),
                        (196504, 0.0),
                        (196505, 0.0),
                    ),
                ),
                (
                    1,
                    (
                        (196412, 0.1),
                        (196501, 0.1),
                        (196502, 0.1),
                        (196503, 0.1),
                        (196504, 0.1),
                        (196505, 0.1),
                    ),
                ),
            ),
        )

    def test_frame_from_tsv_d(self) -> None:
        f1 = sf.Frame.from_elements([1], columns=['a'])

        with temp_file('.txt', path=True) as fp:
            f1.to_tsv(fp)
            f2 = sf.Frame.from_tsv(fp, index_depth=1)
            self.assertEqual(f2.to_pairs(), (('a', ((0, 1),)),))

    def test_frame_from_tsv_e(self) -> None:
        f1 = sf.Frame.from_elements([1], columns=['with space'])

        with temp_file('.txt', path=True) as fp:
            f1.to_tsv(fp)
            f2 = sf.Frame.from_tsv(fp, index_depth=1)
            self.assertEqual(f2.columns.values.tolist(), ['with space'])

    def test_frame_from_tsv_f(self) -> None:
        f1 = sf.Frame.from_elements([1], columns=[':with:colon:'])

        with temp_file('.txt', path=True) as fp:
            f1.to_tsv(fp)
            f2 = sf.Frame.from_tsv(fp, index_depth=1)
            self.assertEqual(f2.to_pairs(), ((':with:colon:', ((0, 1),)),))

    def test_frame_from_tsv_g(self) -> None:
        f1 = sf.Frame.from_elements(
            ['#', '*', '@'], columns=['a', '#', 'c'], index=('q', 'r', 's')
        )

        with temp_file('.txt', path=True) as fp:
            f1.to_tsv(fp)
            f2 = sf.Frame.from_tsv(fp, index_depth=1)
            self.assertEqualFrames(f1, f2)

    def test_frame_from_tsv_h(self) -> None:
        with temp_file('.txt', path=True) as fp:
            with open(fp, 'w', encoding='utf-8') as file:
                file.write('\n'.join(('index\tA\tB', 'a\tTrue\t20.2', 'b\tFalse\t85.3')))
                file.close()

            f1 = sf.Frame.from_tsv(
                fp,
                index_depth=1,
                columns_depth=1,
                dtypes=(None, bool, float),  # position dtypes include index
            )
            self.assertEqual(
                f1.astype['B'](str).to_pairs(),
                (
                    ('A', (('a', True), ('b', False))),
                    ('B', (('a', '20.2'), ('b', '85.3'))),
                ),
            )

    def test_frame_from_tsv_i(self) -> None:
        f1 = Frame.from_element(1, index=Index([1], name='foo'), columns=['a'])

        with temp_file('.txt', path=True) as fp:
            f1.to_tsv(fp)  # this writes the index
            f2 = Frame.from_tsv(fp, index_depth=1, index_name_depth_level=-1)
            self.assertEqual(f2.index.name, 'foo')

            # provide a list means that we want each label to be atuple
            f3 = Frame.from_tsv(fp, index_depth=1, index_name_depth_level=[0])
            self.assertEqual(f3.index.name, ('foo',))

    def test_frame_from_tsv_j(self) -> None:
        f1 = Frame.from_element(1, index=Index([1], name='foo'), columns=['a'])

        with temp_file('.txt', path=True) as fp:
            f1.to_tsv(fp)  # this writes the index

            f2 = Frame.from_tsv(fp, index_depth=1, columns_name_depth_level=-1)
            self.assertEqual(f2.columns.name, 'foo')
            self.assertEqual(f2.index.name, None)

            f3 = Frame.from_tsv(fp, index_depth=0, columns_name_depth_level=-1)

            self.assertEqual(f3.index.name, None)

    def test_frame_from_tsv_k(self) -> None:
        index = IndexHierarchy.from_labels((('a', 1), ('a', 2)), name=('foo', 'bar'))
        f1 = Frame(np.arange(4).reshape(2, 2), index=index, columns=('x', 'y'))

        with temp_file('.txt', path=True) as fp:
            f1.to_tsv(fp)  # this writes the index

            f2 = Frame.from_tsv(fp, index_depth=2, index_name_depth_level=-1)

            self.assertEqual(f2.index.name, ('foo', 'bar'))

    def test_frame_from_tsv_m(self) -> None:
        index = IndexHierarchy.from_labels((('a', 1), ('a', 2)), name=('up', 'down'))
        columns = IndexHierarchy.from_product(('x', 'y'), (10, 20))

        f1 = Frame(np.arange(8).reshape(2, 4), index=index, columns=columns)

        with temp_file('.txt', path=True) as fp:
            f1.to_tsv(fp)  # this writes the index

            f2 = Frame.from_tsv(
                fp,
                index_depth=2,
                index_name_depth_level=0,
                columns_depth=2,
            )
            self.assertEqual(f2.index.name, ('up', 'down'))
            self.assertEqual(f2.columns.name, None)

    def test_frame_from_tsv_n(self) -> None:
        f1 = sf.Frame(columns=['column'], index=sf.Index([], name='index'))
        s = io.StringIO()
        f1.to_tsv(s)

        s.seek(0)
        f2 = sf.Frame.from_tsv(s, index_depth=1, index_name_depth_level=0)
        self.assertEqual(f2.index.name, 'index')
        self.assertEqual(f2.to_pairs(), (('column', ()),))

        s.seek(0)
        f3 = sf.Frame.from_tsv(s, index_depth=1, columns_name_depth_level=0)
        self.assertEqual(f3.columns.name, 'index')
        self.assertEqual(f3.to_pairs(), (('column', ()),))

    # ---------------------------------------------------------------------------

    def test_frame_to_pairs_a(self) -> None:
        records = (
            (2, 'a'),
            (3, 'b'),
        )
        f1 = Frame.from_records(records, columns=('r', 's'), index=('w', 'x'))

        with self.assertRaises(AxisInvalid):
            x = f1.to_pairs(axis=3)

        post = f1.to_pairs(axis=1)
        self.assertEqual(
            post, (('w', (('r', 2), ('s', 'a'))), ('x', (('r', 3), ('s', 'b'))))
        )

    # ---------------------------------------------------------------------------

    def test_frame_to_str_records_a(self) -> None:
        records = ((2, None), (3, np.nan), (0, False), (3, 'x'))
        f1 = Frame.from_records(
            records,
            columns=('r', 's'),
            index=IndexHierarchy.from_product((1, 2), ('a', 'b')),
        )

        self.assertEqual(
            tuple(f1._to_str_records()),
            (
                ['__index0__', '__index1__', 'r', 's'],
                ['1', 'a', '2', 'None'],
                ['1', 'b', '3', ''],
                ['2', 'a', '0', 'False'],
                ['2', 'b', '3', 'x'],
            ),
        )

        self.assertEqual(
            tuple(f1._to_str_records(include_index_name=False)),
            (
                ['', '', 'r', 's'],
                ['1', 'a', '2', 'None'],
                ['1', 'b', '3', ''],
                ['2', 'a', '0', 'False'],
                ['2', 'b', '3', 'x'],
            ),
        )

        self.assertEqual(
            tuple(f1._to_str_records(include_index=False)),
            (['r', 's'], ['2', 'None'], ['3', ''], ['0', 'False'], ['3', 'x']),
        )
        self.assertEqual(
            tuple(f1._to_str_records(include_index=False, include_columns=False)),
            (['2', 'None'], ['3', ''], ['0', 'False'], ['3', 'x']),
        )

        with self.assertRaises(RuntimeError):
            _ = tuple(
                f1._to_str_records(include_index_name=True, include_columns_name=True)
            )

    # ---------------------------------------------------------------------------
    @skip_win
    def test_frame_to_delimited_a(self) -> None:
        records = (
            (2, None),
            (3, np.nan),
        )
        f1 = Frame.from_records(records, columns=('r', 's'), index=('w', 'x'))

        with temp_file('.txt', path=True) as fp:
            f1.to_delimited(fp, delimiter='|', store_filter=None)
            with open(fp, encoding='utf-8') as f:
                lines = f.readlines()
            self.assertEqual(lines, ['__index0__|r|s\n', 'w|2|None\n', 'x|3|nan\n'])

    @skip_win
    def test_frame_to_delimited_b(self) -> None:
        records = ((2, None), (3, np.nan), (0, False), (3, 'x'))
        f1 = Frame.from_records(
            records,
            columns=('r', 's'),
            index=IndexHierarchy.from_product((1, 2), ('a', 'b')),
        )

        with temp_file('.txt', path=True) as fp:
            f1.to_delimited(fp, delimiter='|', store_filter=None)
            with open(fp, encoding='utf-8') as f:
                lines = f.readlines()
            self.assertEqual(
                lines,
                [
                    '__index0__|__index1__|r|s\n',
                    '1|a|2|None\n',
                    '1|b|3|nan\n',
                    '2|a|0|False\n',
                    '2|b|3|x\n',
                ],
            )

    @skip_win
    def test_frame_to_delimited_c(self) -> None:
        records = (
            (False, 0.000000020, 0.000000123),
            (True, 0.000001119, np.nan),
        )
        f1 = Frame.from_records(records, columns=('r', 's', 't'), index=('w', 'x'))

        sf1 = StoreFilter(
            value_format_float_positional='{:.8f}',
            value_format_float_scientific='{:.8f}',
        )
        sf2 = StoreFilter(
            value_format_float_positional='{:.4e}',
            value_format_float_scientific='{:.4e}',
        )
        with temp_file('.txt', path=True) as fp:
            f1.to_delimited(fp, delimiter='|', store_filter=sf1)
            with open(fp, encoding='utf-8') as f:
                lines1 = f.readlines()
            self.assertEqual(
                lines1,
                [
                    '__index0__|r|s|t\n',
                    'w|False|0.00000002|0.00000012\n',
                    'x|True|0.00000112|\n',
                ],
            )

        with temp_file('.txt', path=True) as fp:
            f1.to_delimited(fp, delimiter='|', store_filter=sf2)
            with open(fp, encoding='utf-8') as f:
                lines2 = f.readlines()
            self.assertEqual(
                lines2,
                [
                    '__index0__|r|s|t\n',
                    'w|False|2.0000e-08|1.2300e-07\n',
                    'x|True|1.1190e-06|\n',
                ],
            )

    @skip_win
    def test_frame_to_delimited_d(self) -> None:
        records = ((2, None), (3, 'a'), (0, 'b'), (3, 'x'))
        f1 = Frame.from_records(
            records,
            columns=('r', 's'),
            index=IndexHierarchy.from_product((1, 2), ('a', 'b'), name=('foo', 'bar')),
        )

        with temp_file('.txt', path=True) as fp1:
            f1.to_delimited(fp1, delimiter='|', store_filter=None)
            with open(fp1, encoding='utf-8') as f:
                lines = f.readlines()
            self.assertEqual(
                lines,
                [
                    'foo|bar|r|s\n',
                    '1|a|2|None\n',
                    '1|b|3|a\n',
                    '2|a|0|b\n',
                    '2|b|3|x\n',
                ],
            )

        with temp_file('.txt', path=True) as fp2:
            f1.to_delimited(
                fp2, delimiter='|', store_filter=None, include_index_name=False
            )
            with open(fp2, encoding='utf-8') as f:
                lines = f.readlines()
            self.assertEqual(
                lines,
                ['||r|s\n', '1|a|2|None\n', '1|b|3|a\n', '2|a|0|b\n', '2|b|3|x\n'],
            )

    @skip_win
    def test_frame_to_delimited_e(self) -> None:
        records = (
            (2, None, 20, False),
            (3, 'a', 30, True),
        )
        f1 = Frame.from_records(
            records,
            index=('r', 's'),
            columns=IndexHierarchy.from_product((1, 2), ('a', 'b'), name=('foo', 'bar')),
        )

        with temp_file('.txt', path=True) as fp1:
            f1.to_delimited(
                fp1, delimiter='|', store_filter=None, include_index_name=False
            )
            with open(fp1, encoding='utf-8') as f:
                lines = f.readlines()
            self.assertEqual(
                lines,
                ['|1|1|2|2\n', '|a|b|a|b\n', 'r|2|None|20|False\n', 's|3|a|30|True\n'],
            )

        with temp_file('.txt', path=True) as fp2:
            f1.to_delimited(
                fp2,
                delimiter='|',
                store_filter=None,
                include_index_name=False,
                include_columns_name=True,
            )
            with open(fp2, encoding='utf-8') as f:
                lines = f.readlines()

            self.assertEqual(
                lines,
                [
                    'foo|1|1|2|2\n',
                    'bar|a|b|a|b\n',
                    'r|2|None|20|False\n',
                    's|3|a|30|True\n',
                ],
            )

        with temp_file('.txt', path=True) as fp3:
            with self.assertRaises(RuntimeError):
                f1.to_delimited(
                    fp3,
                    delimiter='|',
                    store_filter=None,
                    include_index_name=True,
                    include_columns_name=True,
                )

    # ---------------------------------------------------------------------------

    def test_frame_to_csv_a(self) -> None:
        records = (
            (2, 2, 'a', False, False),
            (30, 34, 'b', True, False),
            (2, 95, 'c', False, False),
            (30, 73, 'd', True, True),
        )
        f1 = Frame.from_records(
            records, columns=('p', 'q', 'r', 's', 't'), index=('w', 'x', 'y', 'z')
        )

        file = StringIO()
        f1.to_csv(file)
        file.seek(0)
        self.assertEqual(
            file.read(),
            '__index0__,p,q,r,s,t\nw,2,2,a,False,False\nx,30,34,b,True,False\ny,2,95,c,False,False\nz,30,73,d,True,True\n',
        )

        file = StringIO()
        f1.to_csv(file, include_index=False)
        file.seek(0)
        self.assertEqual(
            file.read(),
            'p,q,r,s,t\n2,2,a,False,False\n30,34,b,True,False\n2,95,c,False,False\n30,73,d,True,True\n',
        )

        file = StringIO()
        f1.to_csv(file, include_index=False, include_columns=False)
        file.seek(0)
        self.assertEqual(
            file.read(),
            '2,2,a,False,False\n30,34,b,True,False\n2,95,c,False,False\n30,73,d,True,True\n',
        )

    def test_frame_to_csv_b(self) -> None:
        f = sf.Frame.from_elements(
            [1, 2, 3], columns=['a'], index=sf.Index(range(3), name='Important Name')
        )
        file = StringIO()
        f.to_csv(file)
        file.seek(0)
        self.assertEqual(file.read(), 'Important Name,a\n0,1\n1,2\n2,3\n')

    def test_frame_to_csv_c(self) -> None:
        records = (
            (2, np.nan, 'a', False, None),
            (30, np.nan, 'b', True, None),
            (2, np.inf, 'c', False, None),
            (30, -np.inf, 'd', True, None),
        )
        f1 = Frame.from_records(
            records, columns=('p', 'q', 'r', 's', 't'), index=('w', 'x', 'y', 'z')
        )

        with temp_file('.csv') as fp:
            f1.to_csv(fp)

            with open(fp, encoding='utf-8') as f:
                lines = f.readlines()
                # nan has been converted to string
                self.assertEqual(lines[1], 'w,2,,a,False,None\n')
                self.assertEqual(lines[4], 'z,30,-inf,d,True,None\n')

    def test_frame_to_csv_d(self) -> None:
        f1 = Frame.from_records(
            ((10, 20, 50, 60), (50.0, 60.4, -50, -60)),
            index=('p', 'q'),
            columns=IndexHierarchy.from_product(('I', 'II'), ('a', 'b')),
            name='f3',
        )

        with temp_file('.csv') as fp:
            f1.to_csv(fp)

            with open(fp, encoding='utf-8') as f:
                lines = f.readlines()

            self.assertEqual(
                lines,
                [
                    '__index0__,I,I,II,II\n',
                    ',a,b,a,b\n',
                    'p,10.0,20.0,50,60\n',
                    'q,50.0,60.4,-50,-60\n',
                ],
            )

            f2 = Frame.from_csv(fp, columns_depth=2, index_depth=1)
            self.assertEqual(
                f2.to_pairs(),
                (
                    (('I', 'a'), (('p', 10.0), ('q', 50.0))),
                    (('I', 'b'), (('p', 20.0), ('q', 60.4))),
                    (('II', 'a'), (('p', 50), ('q', -50))),
                    (('II', 'b'), (('p', 60), ('q', -60))),
                ),
            )

    def test_frame_to_csv_e(self) -> None:
        f1 = Frame.from_records(
            ((10, 20, 50, 60), (50.0, 60.4, -50, -60)),
            index=('p', 'q'),
            columns=IndexHierarchy.from_product(
                (10, 20),
                ('I', 'II'),
            ),
            name='f3',
        )

        with temp_file('.csv') as fp:
            f1.to_csv(fp)

            with open(fp, encoding='utf-8') as f:
                lines = f.readlines()

            self.assertEqual(
                lines,
                [
                    '__index0__,10,10,20,20\n',
                    ',I,II,I,II\n',
                    'p,10.0,20.0,50,60\n',
                    'q,50.0,60.4,-50,-60\n',
                ],
            )
            f2 = Frame.from_csv(fp, columns_depth=2, index_depth=1)
            self.assertEqual(
                f2.to_pairs(),
                (
                    ((10, 'I'), (('p', 10.0), ('q', 50.0))),
                    ((10, 'II'), (('p', 20.0), ('q', 60.4))),
                    ((20, 'I'), (('p', 50), ('q', -50))),
                    ((20, 'II'), (('p', 60), ('q', -60))),
                ),
            )

    # ---------------------------------------------------------------------------

    def test_frame_to_tsv_a(self) -> None:
        records = (
            (2, 2, 'a', False, False),
            (30, 34, 'b', True, False),
            (2, 95, 'c', False, False),
            (30, 73, 'd', True, True),
        )
        f1 = Frame.from_records(
            records, columns=('p', 'q', 'r', 's', 't'), index=('w', 'x', 'y', 'z')
        )

        file = StringIO()
        f1.to_tsv(file)
        file.seek(0)
        self.assertEqual(
            file.read(),
            '__index0__\tp\tq\tr\ts\tt\nw\t2\t2\ta\tFalse\tFalse\nx\t30\t34\tb\tTrue\tFalse\ny\t2\t95\tc\tFalse\tFalse\nz\t30\t73\td\tTrue\tTrue\n',
        )

    def test_frame_to_tsv_b(self) -> None:
        records = (
            (2, 2, 'a', False, False),
            (30, 34, 'b', True, False),
            (2, 95, 'c', False, False),
            (30, 73, 'd', True, True),
        )

        index = IndexHierarchy.from_product(
            ('A', 'B'),
            (1, 2),
            index_constructors=(Index, partial(Index, dtype=np.int64)),
        )

        f1 = Frame.from_records(
            records,
            columns=('p', 'q', 'r', 's', 't'),
            index=index,
            dtypes={'p': np.int64, 'q': np.int64},
        )
        with temp_file('.txt', path=True) as fp:
            f1.to_tsv(fp, include_index=True)
            f2 = Frame.from_tsv(fp, index_depth=2)
            self.assertEqualFrames(f1, f2)

    def test_frame_to_tsv_c(self) -> None:
        columns = IndexHierarchy.from_product(
            ('III', 'IV'),
            (10, 20),
            index_constructors=(Index, partial(Index, dtype=np.int64)),
        )

        f1 = sf.Frame(
            np.arange(16, dtype=np.int64).reshape((4, 4)),
            index=sf.IndexHierarchy.from_product(('I', 'II'), ('a', 'b')),
            columns=columns,
        )

        with temp_file('.txt', path=True) as fp:
            f1.to_tsv(fp, include_index=True)
            f2 = Frame.from_tsv(fp, index_depth=2, columns_depth=2)
            self.assertEqualFrames(f1, f2)

    def test_frame_to_tsv_d(self) -> None:
        f1 = ff.parse('s(4,5)')

        with temp_file('', path=True) as fp:
            fp = os.path.join(fp, '__space__', 'test.txt')
            with self.assertRaises((NotADirectoryError, FileNotFoundError)):
                f1.to_tsv(fp)

    # ---------------------------------------------------------------------------

    def test_frame_to_html_a(self) -> None:
        records = (
            (2, 'a', False),
            (3, 'b', False),
        )
        f1 = Frame.from_records(records, columns=('r', 's', 't'), index=('w', 'x'))
        post = f1.to_html(style_config=None)

        self.assertEqual(
            post,
            '<table><thead><tr><th></th><th>r</th><th>s</th><th>t</th></tr></thead><tbody><tr><th>w</th><td>2</td><td>a</td><td>False</td></tr><tr><th>x</th><td>3</td><td>b</td><td>False</td></tr></tbody></table>',
        )

        msg = str(f1.display(sf.DisplayConfig(type_show=False, include_columns=False)))
        self.assertEqual(msg, 'w 2 a False\nx 3 b False')

    def test_frame_to_html_datatables_a(self) -> None:
        records = (
            (2, 'a', False),
            (3, 'b', False),
        )
        f1 = Frame.from_records(records, columns=('r', 's', 't'), index=('w', 'x'))

        sio = StringIO()

        post = f1.to_html_datatables(sio, show=False)

        self.assertEqual(post, None)
        self.assertTrue(len(sio.read()) > 1300)

    # ---------------------------------------------------------------------------

    def test_frame_to_rst_a(self) -> None:
        records = (
            (2, 'a', False),
            (3, 'b', False),
        )
        f1 = Frame.from_records(records, columns=('r', 's', 't'), index=('w', 'x'))
        post = f1.to_rst()
        msg = """
                +--+--+--+-----+
                |  |r |s |t    |
                +==+==+==+=====+
                |w |2 |a |False|
                +--+--+--+-----+
                |x |3 |b |False|
                +--+--+--+-----+
                """
        self.assertEqualLines(post, msg)

    def test_frame_to_markdown_a(self) -> None:
        records = (
            (2, 'a', False),
            (3, 'b', False),
        )
        f1 = Frame.from_records(records, columns=('r', 's', 't'), index=('w', 'x'))
        post = f1.to_markdown()

        msg = """
                |  |r |s |t    |
                |--|--|--|-----|
                |w |2 |a |False|
                |x |3 |b |False|
                """
        self.assertEqualLines(post, msg)

    def test_frame_to_latex_a(self) -> None:
        records = (
            (2, 'a', False),
            (3, 'b', False),
        )
        f1 = Frame.from_records(records, columns=('r', 's', 't'), index=('w', 'x'))
        post = f1.to_latex()
        msg = r"""
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
                """
        self.assertEqualLines(post, msg)

    # ---------------------------------------------------------------------------

    def test_frame_to_xlsx_a(self) -> None:
        records = (
            (2, 2, 'a', False, False),
            (30, 34, 'b', True, False),
            (2, 95, 'c', False, False),
            (30, 73, 'd', True, True),
        )
        f1 = Frame.from_records(
            records, columns=('p', 'q', 'r', 's', 't'), index=('w', 'x', 'y', 'z')
        )

        config = StoreConfig(index_depth=1)

        with temp_file('.xlsx') as fp:
            f1.to_xlsx(fp)
            st = StoreXLSX(fp)
            f2 = st.read(label=STORE_LABEL_DEFAULT, config=config)
            self.assertEqualFrames(f1, f2)

    def test_frame_to_xlsx_b(self) -> None:
        records = (
            (2, 2, 'a', False, False),
            (30, 34, 'b', True, False),
            (2, 95, 'c', False, False),
            (30, 73, 'd', True, True),
        )
        f1 = Frame.from_records(
            records,
            columns=('p', 'q', 'r', 's', 't'),
            index=Index(('w', 'x', 'y', 'z'), name='foo'),
            name='bar',
        )

        with temp_file('.xlsx') as fp:
            f1.to_xlsx(fp)
            f2 = Frame.from_xlsx(fp, name='baz')
            self.assertEqual(f2.name, 'baz')
            self.assertEqual(f2.columns.values.tolist(), ['foo', 'p', 'q', 'r', 's', 't'])

    def test_frame_to_xlsx_c(self) -> None:
        records = (
            (2, 2, 'a', False, False),
            (30, 34, 'b', True, False),
            (2, 95, 'c', False, False),
            (30, 73, 'd', True, True),
        )
        f1 = Frame.from_records(
            records,
            columns=('p', 'q', 'r', 's', 't'),
            index=IndexHierarchy.from_product(('a', 'b'), (1, 2), name=('foo', 'bar')),
        )

        with temp_file('.xlsx') as fp:
            with self.assertRaises(RuntimeError):
                f1.to_xlsx(fp, include_index_name=True, include_columns_name=True)

            f1.to_xlsx(fp)
            f2 = Frame.from_xlsx(fp)
            self.assertEqual(
                f2.columns.values.tolist(), ['foo', 'bar', 'p', 'q', 'r', 's', 't']
            )

            f3 = Frame.from_xlsx(fp, index_depth=2, index_name_depth_level=0)
            self.assertEqual(f3.index.name, ('foo', 'bar'))

            f4 = Frame.from_xlsx(fp, index_depth=2, columns_name_depth_level=0)
            self.assertEqual(f4.columns.name, 'foo')

            f5 = Frame.from_xlsx(fp, index_depth=2, columns_name_depth_level=(0, 1))
            self.assertEqual(f5.columns.name, ('foo', 'bar'))

    def test_frame_to_xlsx_d(self) -> None:
        records = (
            (2, 2, 'a', False),
            (30, 34, 'b', True),
            (2, 95, 'c', False),
            (30, 73, 'd', True),
        )
        f1 = Frame.from_records(
            records,
            columns=IndexHierarchy.from_product(('a', 'b'), (1, 2), name=('foo', 'bar')),
            index=('p', 'q', 'r', 's'),
        )

        with temp_file('.xlsx') as fp:
            f1.to_xlsx(fp, include_index_name=False, include_columns_name=True)
            f2 = Frame.from_xlsx(fp, columns_depth=2)

            # loads labels over index in as first column header
            self.assertEqual(
                f2.to_pairs(),
                (
                    (('foo', 'bar'), ((0, 'p'), (1, 'q'), (2, 'r'), (3, 's'))),
                    (('a', 1), ((0, 2), (1, 30), (2, 2), (3, 30))),
                    (('a', 2), ((0, 2), (1, 34), (2, 95), (3, 73))),
                    (('b', 1), ((0, 'a'), (1, 'b'), (2, 'c'), (3, 'd'))),
                    (('b', 2), ((0, False), (1, True), (2, False), (3, True))),
                ),
            )

            f3 = Frame.from_xlsx(
                fp, columns_depth=2, index_depth=1, columns_name_depth_level=0
            )
            self.assertEqual(f3.columns.name, ('foo', 'bar'))

            f4 = Frame.from_xlsx(
                fp, columns_depth=2, index_depth=1, index_name_depth_level=(0, 1)
            )
            self.assertEqual(f4.index.name, ('foo', 'bar'))

    def test_frame_to_xlsx_e(self) -> None:
        f1 = Frame.from_element(0.0, columns=(0.0,), index=(0.0,))

        with temp_file('.xlsx') as fp:
            f1.to_xlsx(fp)
            f2 = Frame.from_xlsx(fp)
            self.assertEqual(f2.columns.values.tolist(), ['__index0__', 0])

    def test_frame_to_xlsx_f(self) -> None:
        f1 = Frame.from_element('', columns=('a', 'b'), index=('x', 'y'))

        with temp_file('.xlsx') as fp:
            f1.to_xlsx(fp)
            f2 = Frame.from_xlsx(fp, index_depth=1)
            self.assertEqual(
                f2.fillna(0).to_pairs(),
                (('a', (('x', 0.0), ('y', 0.0))), ('b', (('x', 0.0), ('y', 0.0)))),
            )

            f3 = Frame.from_xlsx(
                fp, index_depth=1, store_filter=StoreFilter(to_nan=frozenset())
            )
            self.assertEqual(
                f3.to_pairs(),
                (('a', (('x', ''), ('y', ''))), ('b', (('x', ''), ('y', '')))),
            )

    def test_frame_to_xlsx_g(self) -> None:
        f1 = sf.Frame.from_dict(
            dict(a=[8, 8, 1, 3, 3, 2, 2], b=list('ababacd'), c=[0, 1, 2, 3, 4, 5, 6])
        ).set_index_hierarchy(['a', 'b'])

        lwd = tuple(f1.index.label_widths_at_depth(0))
        self.assertEqual(lwd, ((8, 2), (1, 1), (3, 2), (2, 2)))

        with temp_file('.xlsx') as fp:
            f1.to_xlsx(fp)
            f2 = sf.Frame.from_xlsx(fp, index_depth=2)
            self.assertTrue(f1.index.equals(f2.index))

    # ---------------------------------------------------------------------------

    def test_frame_from_xlsx_a(self) -> None:
        records = (
            (2, 2, 'a', False, False),
            (30, 34, 'b', True, False),
            (2, 95, 'c', False, False),
            (30, 73, 'd', True, True),
        )
        f1 = Frame.from_records(
            records, columns=('p', 'q', 'r', 's', 't'), index=('w', 'x', 'y', 'z')
        )

        with temp_file('.xlsx') as fp:
            f1.to_xlsx(fp)
            f2 = Frame.from_xlsx(fp, index_depth=f1.index.depth)
            self.assertEqualFrames(f1, f2)

            f3 = FrameGO.from_xlsx(fp, index_depth=f1.index.depth)
            self.assertEqual(f3.__class__, FrameGO)
            self.assertEqual(f3.shape, (4, 5))

    def test_frame_from_xlsx_b(self) -> None:
        f1 = Frame.from_records(
            (
                (10, 20, 50, False, 10, 20, 50, False),
                (50.0, 60.4, -50, True, 50.0, 60.4, -50, True),
                (234, 44452, 0, False, 234, 44452, 0, False),
                (4, -4, 2000, True, 4, -4, 2000, True),
                (10, 20, 50, False, 10, 20, 50, False),
                (50.0, 60.4, -50, True, 50.0, 60.4, -50, True),
                (234, 44452, 0, False, 234, 44452, 0, False),
                (4, -4, 2000, True, 4, -4, 2000, True),
            ),
            index=IndexHierarchy.from_product(
                ('top', 'bottom'), ('far', 'near'), ('left', 'right')
            ),
            columns=IndexHierarchy.from_product(('I', 'II'), ('a', 'b'), (1, 2)),
        )

        with temp_file('.xlsx') as fp:
            f1.to_xlsx(fp)
            f2 = Frame.from_xlsx(
                fp, index_depth=f1.index.depth, columns_depth=f1.columns.depth
            )
            self.assertEqualFrames(f1, f2, compare_dtype=False)

    def test_frame_from_xlsx_d(self) -> None:
        # isolate case of all None data that has a valid index

        f1 = Frame.from_element(None, index=('a', 'b', 'c'), columns=('x', 'y', 'z'))

        with temp_file('.xlsx') as fp:
            f1.to_xlsx(fp)
            f2 = Frame.from_xlsx(
                fp, index_depth=f1.index.depth, columns_depth=f1.columns.depth
            )
            self.assertEqualFrames(f1, f2)

    def test_frame_from_xlsx_e(self) -> None:
        # isolate case of all None data that has a valid IndexHierarchy

        f1 = Frame.from_element(
            None,
            index=IndexHierarchy.from_product((0, 1), ('a', 'b')),
            columns=('x', 'y', 'z'),
        )

        with temp_file('.xlsx') as fp:
            f1.to_xlsx(fp)
            f2 = Frame.from_xlsx(
                fp, index_depth=f1.index.depth, columns_depth=f1.columns.depth
            )
            self.assertEqualFrames(f1, f2)

    def test_frame_from_xlsx_f1(self) -> None:
        # isolate case of all None data and only columns
        f1 = Frame.from_element(None, index=('a', 'b', 'c'), columns=('x', 'y', 'z'))

        with temp_file('.xlsx') as fp:
            f1.to_xlsx(fp, include_index=False)
            f2 = Frame.from_xlsx(
                fp,
                index_depth=0,
                columns_depth=f1.columns.depth,
                trim_nadir=True,
            )
        # drop all rows, keeps columns as we gave columns depth
        self.assertEqual(f2.shape, (0, 3))

    def test_frame_from_xlsx_f2(self) -> None:
        # isolate case of all None data and only columns
        f1 = Frame.from_element(None, index=('a', 'b', 'c'), columns=('x', 'y', 'z'))

        with temp_file('.xlsx') as fp:
            f1.to_xlsx(fp, include_index=False)
            f2 = Frame.from_xlsx(
                fp,
                index_depth=0,
                columns_depth=f1.columns.depth,
                trim_nadir=False,
            )
        self.assertEqual(f2.shape, (3, 3))
        self.assertEqual(
            f2.to_pairs(),
            (
                ('x', ((0, None), (1, None), (2, None))),
                ('y', ((0, None), (1, None), (2, None))),
                ('z', ((0, None), (1, None), (2, None))),
            ),
        )

    def test_frame_from_xlsx_g1(self) -> None:
        # isolate case of all None data, no index, no columns
        f1 = Frame.from_element(None, index=('a', 'b', 'c'), columns=('x', 'y', 'z'))

        with temp_file('.xlsx') as fp:
            f1.to_xlsx(fp, include_index=False, include_columns=False)
            with self.assertRaises(ErrorInitFrame):
                f2 = Frame.from_xlsx(
                    fp,
                    index_depth=0,
                    columns_depth=0,
                    trim_nadir=True,
                )

    def test_frame_from_xlsx_g2(self) -> None:
        # isolate case of all None data, no index, no columns
        f1 = Frame.from_element(None, index=('a', 'b', 'c'), columns=('x', 'y', 'z'))

        with temp_file('.xlsx') as fp:
            f1.to_xlsx(fp, include_index=False, include_columns=False)
            f2 = Frame.from_xlsx(
                fp,
                index_depth=0,
                columns_depth=0,
                trim_nadir=False,
            )

        self.assertEqual(
            f2.to_pairs(),
            (
                (0, ((0, None), (1, None), (2, None))),
                (1, ((0, None), (1, None), (2, None))),
                (2, ((0, None), (1, None), (2, None))),
            ),
        )

    def test_frame_from_xlsx_h(self) -> None:
        records = (
            (2, 2, 'a', False, False),
            (30, 34, 'b', True, False),
            (2, 95, 'c', 'k', 'r'),
            (30, 73, 'd', True, True),
            (2, 95, 'c', False, False),
            (30, 73, 'd', True, True),
        )
        f1 = Frame.from_records(
            records,
            columns=('p', 'q', 'r', 's', 't'),
            index=('u', 'v', 'w', 'x', 'y', 'z'),
        )

        with temp_file('.xlsx') as fp:
            f1.to_xlsx(fp)
            f2 = Frame.from_xlsx(
                fp,
                index_depth=1,
                skip_header=3,  # include the column that was added
                skip_footer=1,
            )

        self.assertEqual(f2.shape, (2, 5))
        self.assertEqual(f2.columns.values.tolist(), [2, 95, 'c', 'k', 'r'])
        self.assertEqual(f2.index.values.tolist(), ['x', 'y'])

    def test_frame_from_xlsx_i(self) -> None:
        records = (
            (2, 2, 'a', False, False),
            (30, 34, 'b', True, False),
            (2, 95, 'c', 'k', 'r'),
            (30, 73, 'd', True, True),
            (2, 95, 'c', False, False),
            (30, 73, 'd', True, True),
        )
        f1 = Frame.from_records(
            records,
            columns=('p', 'q', 'r', 's', 't'),
            index=('u', 'v', 'w', 'x', 'y', 'z'),
        )

        # for an unexplained reason, openpyxl raises a ResourceWarning here
        with temp_file('.xlsx') as fp, WarningsSilent():
            f1.to_xlsx(fp)
            f2 = Frame.from_xlsx(
                fp,
                columns_depth=0,
                skip_header=4,  # include the column that was added
                skip_footer=2,
            )

        self.assertEqual(f2.shape, (1, 6))
        self.assertEqual(f2.columns.values.tolist(), list(range(6)))
        self.assertEqual(f2.index.values.tolist(), [0])

    def test_frame_from_xlsx_j(self) -> None:
        records = (
            (2, 2, 'a', False, False),
            (None, None, None, None, None),
            (None, None, None, None, None),
            (30, 73, 'd', True, True),
            (2, 95, 'c', False, False),
            (30, 73, 'd', True, True),
        )
        f1 = Frame.from_records(records, columns=('p', 'q', 'r', 's', 't'))

        with temp_file('.xlsx') as fp:
            f1.to_xlsx(fp, include_index=False)

            f2 = Frame.from_xlsx(
                fp,
                columns_depth=1,
                index_depth=0,
                trim_nadir=True,
            )
            self.assertEqual(f2.shape, (6, 5))
            self.assertEqual(
                f2.to_pairs(),
                (
                    ('p', ((0, 2), (1, None), (2, None), (3, 30), (4, 2), (5, 30))),
                    ('q', ((0, 2), (1, None), (2, None), (3, 73), (4, 95), (5, 73))),
                    (
                        'r',
                        ((0, 'a'), (1, None), (2, None), (3, 'd'), (4, 'c'), (5, 'd')),
                    ),
                    (
                        's',
                        (
                            (0, False),
                            (1, None),
                            (2, None),
                            (3, True),
                            (4, False),
                            (5, True),
                        ),
                    ),
                    (
                        't',
                        (
                            (0, False),
                            (1, None),
                            (2, None),
                            (3, True),
                            (4, False),
                            (5, True),
                        ),
                    ),
                ),
            )

    def test_frame_from_xlsx_k(self) -> None:
        records = (
            (2, 2, 'a', False, None),
            (None, None, None, None, None),
            (None, None, None, None, None),
            (30, 73, 'd', True, None),
            (None, None, None, None, None),
            (None, None, None, None, None),
        )
        f1 = Frame.from_records(records, columns=('p', 'q', 'r', 's', 't'))

        with temp_file('.xlsx') as fp:
            f1.to_xlsx(fp, include_index=False)

            f2 = Frame.from_xlsx(
                fp,
                columns_depth=1,
                index_depth=0,
                trim_nadir=True,
            )
            # we keep the last column (all None) because there is a valid label
            self.assertEqual(f2.shape, (4, 5))

    def test_frame_from_xlsx_m(self) -> None:
        records = (
            (2, 2, 'a', False, None),
            (None, None, None, None, None),
            (None, None, None, None, None),
            (30, 73, 'd', True, None),
            (None, None, None, None, None),
            (None, None, None, None, None),
        )
        f1 = Frame.from_records(records, columns=('p', 'q', 'r', 's', 't'))

        with temp_file('.xlsx') as fp:
            f1.to_xlsx(fp, include_index=False, include_columns=False)

            f2 = Frame.from_xlsx(
                fp,
                columns_depth=0,
                index_depth=0,
                trim_nadir=True,
            )
            # we keep the last column (all None) because there is a valid label
            self.assertEqual(f2.shape, (4, 4))

    def test_frame_from_xlsx_n(self) -> None:
        records = (
            (2012, '2021-04-17', 'k', False, False),
            (1542, '1945-11-28', 'q', True, False),
        )
        f1 = Frame.from_records(
            records, columns=('p', 'q', 'r', 's', 't'), index=('x', 'y')
        )

        with temp_file('.xlsx') as fp:
            f1.to_xlsx(fp)
            f2 = Frame.from_xlsx(
                fp, index_depth=3, index_constructors=(Index, IndexYear, IndexDate)
            )
            self.assertEqual(f2.index.depth, 3)
            self.assertEqual(
                f2.index.index_types.values.tolist(), [Index, IndexYear, IndexDate]
            )

    def test_frame_from_xlsx_o(self) -> None:
        records = (
            (False, True, 120, 540),
            (True, False, 602, 403),
        )
        f1 = Frame.from_records(
            records,
            columns=IndexHierarchy.from_product(
                (1920, 1542), ('2021-01-05', '1264-10-31')
            ),
            index=('x', 'y'),
        )

        with temp_file('.xlsx') as fp:
            f1.to_xlsx(fp)
            f2 = Frame.from_xlsx(
                fp,
                columns_depth=2,
                index_depth=1,
                columns_constructors=(IndexYear, IndexDate),
            )
            self.assertEqual(f2.columns.depth, 2)
            self.assertEqual(
                f2.columns.index_types.values.tolist(), [IndexYear, IndexDate]
            )
            self.assertEqual(f2.shape, (2, 4))

    # ---------------------------------------------------------------------------

    def test_frame_from_sqlite_a(self) -> None:
        records = (
            (2, 2, 'a', False, False),
            (30, 34, 'b', True, False),
            (2, 95, 'c', False, False),
            (30, 73, 'd', True, True),
        )
        f1 = Frame.from_records(
            records,
            columns=('p', 'q', 'r', 's', 't'),
            index=('w', 'x', 'y', 'z'),
            name='foo',
        )

        with temp_file('.sqlite') as fp:
            f1.to_sqlite(fp)
            f2 = Frame.from_sqlite(fp, label='foo', index_depth=f1.index.depth)
            self.assertEqualFrames(f1, f2)
            self.assertEqual(f2.name, 'foo')

            f3 = FrameGO.from_sqlite(fp, label='foo', index_depth=f1.index.depth)
            self.assertEqual(f3.__class__, FrameGO)
            self.assertEqual(f3.shape, (4, 5))

            f3 = Frame.from_sqlite(
                fp, label='foo', index_depth=f1.index.depth, name='bar'
            )
            self.assertEqual(f3.name, 'bar')

    def test_frame_from_sqlite_b(self) -> None:
        f1 = Frame.from_records(
            (
                (10, 20, 50, False, 10, 20, 50, False),
                (50.0, 60.4, -50, True, 50.0, 60.4, -50, True),
                (234, 44452, 0, False, 234, 44452, 0, False),
                (4, -4, 2000, True, 4, -4, 2000, True),
                (10, 20, 50, False, 10, 20, 50, False),
                (50.0, 60.4, -50, True, 50.0, 60.4, -50, True),
                (234, 44452, 0, False, 234, 44452, 0, False),
                (4, -4, 2000, True, 4, -4, 2000, True),
            ),
            index=IndexHierarchy.from_product(
                ('top', 'bottom'), ('far', 'near'), ('left', 'right')
            ),
            columns=IndexHierarchy.from_product(('I', 'II'), ('a', 'b'), (1, 2)),
            name='foo',
        )

        with temp_file('.sqlite') as fp:
            f1.to_sqlite(fp)
            f2 = Frame.from_sqlite(
                fp,
                label='foo',
                index_depth=f1.index.depth,
                columns_depth=f1.columns.depth,
            )
            self.assertEqualFrames(f1, f2)

    def test_frame_from_sqlite_c(self) -> None:
        records = (
            (2020, '2020-11-12', False, False),
            (2020, '2020-12-31', True, False),
            (1492, '1492-03-12', False, False),
            (1492, '1492-03-19', True, True),
        )
        f1 = Frame.from_records(
            records,
            columns=('q', 'r', 's', 't'),
            index=('w', 'x', 'y', 'z'),
            name='foo',
        )

        with temp_file('.sqlite') as fp:
            f1.to_sqlite(fp, include_index=False)
            f2 = Frame.from_sqlite(
                fp,
                label='foo',
                index_depth=2,
                index_constructors=(IndexYear, IndexDate),
            )
            self.assertEqual(f2.index.depth, 2)
            self.assertEqual(f2.index.index_types.values.tolist(), [IndexYear, IndexDate])

    def test_frame_from_sqlite_d(self) -> None:
        f1 = Frame.from_records(
            (
                (10, 20, 50, False, 10, 20, 50, False),
                (50.0, 60.4, -50, True, 50.0, 60.4, -50, True),
            ),
            columns=IndexHierarchy.from_product(
                ('I', 'II'),
                (1910, 1915),
                ('2021-01-03', '1918-05-04'),
            ),
            name='foo',
        )

        with temp_file('.sqlite') as fp:
            f1.to_sqlite(fp, include_index=False)
            f2 = Frame.from_sqlite(
                fp,
                label='foo',
                index_depth=0,
                columns_depth=3,
                columns_constructors=(Index, IndexYear, IndexDate),
            )
            self.assertEqual(f2.columns.depth, 3)
            self.assertEqual(
                f2.columns.index_types.values.tolist(), [Index, IndexYear, IndexDate]
            )
            self.assertEqual(f2.name, 'foo')

    # ---------------------------------------------------------------------------
    def test_frame_to_sqlite_a(self) -> None:
        f1 = ff.parse('s(4,5)')
        with temp_file('.sqlite') as fp:
            # RuntimeError: must provide a label or define Frame name.
            with self.assertRaises(RuntimeError):
                f1.to_sqlite(fp)

    # ---------------------------------------------------------------------------

    def test_frame_and_a(self) -> None:
        records = (
            (2, 2, 'a', False, False),
            (30, 34, 'b', True, False),
            (2, 95, 'c', False, False),
            (30, 73, 'd', True, True),
        )

        f1 = Frame.from_records(
            records, columns=('p', 'q', 'r', 's', 't'), index=('w', 'x', 'y', 'z')
        )

        self.assertEqual(
            f1.all(axis=0).to_pairs(),
            (('p', True), ('q', True), ('r', True), ('s', False), ('t', False)),
        )

        self.assertEqual(
            f1.any(axis=0).to_pairs(),
            (('p', True), ('q', True), ('r', True), ('s', True), ('t', True)),
        )

        self.assertEqual(
            f1.all(axis=1).to_pairs(),
            (('w', False), ('x', False), ('y', False), ('z', True)),
        )

        self.assertEqual(
            f1.any(axis=1).to_pairs(),
            (('w', True), ('x', True), ('y', True), ('z', True)),
        )

    # ---------------------------------------------------------------------------
    def test_frame_unique_a(self) -> None:
        records = (
            (2, 2, 3.5),
            (30, 34, 60.2),
            (2, 95, 1.2),
            (30, 73, 50.2),
        )
        f1 = Frame.from_records(
            records, columns=('p', 'q', 'r'), index=('w', 'x', 'y', 'z')
        )

        self.assertEqual(
            f1.unique().tolist(), [1.2, 2.0, 3.5, 30.0, 34.0, 50.2, 60.2, 73.0, 95.0]
        )

        records = (
            (2, 2, 2),
            (30, 34, 34),
            (2, 2, 2),
            (30, 73, 73),
        )
        f2 = Frame.from_records(
            records, columns=('p', 'q', 'r'), index=('w', 'x', 'y', 'z')
        )

        self.assertEqual(f2.unique().tolist(), [2, 30, 34, 73])

        self.assertEqual(
            f2.unique(axis=0).tolist(), [[2, 2, 2], [30, 34, 34], [30, 73, 73]]
        )
        self.assertEqual(f2.unique(axis=1).tolist(), [[2, 2], [30, 34], [2, 2], [30, 73]])

    def test_frame_unique_b(self) -> None:
        records = (
            (None, 2, None),
            ('30', 34, '30'),
            (None, 2, None),
            ('30', 34, '30'),
        )
        f1 = Frame.from_records(
            records, columns=('p', 'q', 'r'), index=('w', 'x', 'y', 'z')
        )

        self.assertEqual(len(f1.unique()), 4)

        self.assertEqual(len(f1.unique(axis=0)), 2)

        self.assertEqual(len(f1.unique(axis=1)), 2)

    # ---------------------------------------------------------------------------
    def test_frame_unique_enumerated_a(self) -> None:
        records = (
            (2, False),
            (30, False),
            (2, False),
            (30, True),
        )
        f1 = Frame.from_records(
            records,
            columns=('p', 'q'),
            index=('w', 'x', 'y', 'z'),
        )
        x, y = f1.unique_enumerated()
        self.assertEqual(x.tolist(), [[2, 0], [3, 0], [2, 0], [3, 1]])
        self.assertEqual(y.tolist(), [False, True, 2, 30])

    def test_frame_unique_enumerated_b(self) -> None:
        records = (
            (2, False),
            (30, False),
            (2, None),
            (30, None),
        )
        f1 = Frame.from_records(
            records,
            columns=('p', 'q'),
            index=('w', 'x', 'y', 'z'),
        )

        x, y = f1.unique_enumerated()
        self.assertEqual(x.tolist(), [[0, 2], [1, 2], [0, 3], [1, 3]])
        self.assertEqual(y.tolist(), [2, 30, False, None])

    def test_frame_unique_enumerated_c(self) -> None:
        records = (
            (2, False),
            (30, False),
            (2, None),
            (30, None),
        )
        f1 = Frame.from_records(
            records,
            columns=('p', 'q'),
            index=('w', 'x', 'y', 'z'),
        )

        x, y = f1.unique_enumerated(func=isna_element)
        self.assertEqual(x.tolist(), [[0, 2], [1, 2], [0, -1], [1, -1]])
        self.assertEqual(y.tolist(), [2, 30, False])

    def test_frame_unique_enumerated_d(self) -> None:
        records = (
            (2, 0),
            (30, 0),
            (2, 0),
            (30, 0),
        )
        f1 = Frame.from_records(
            records,
            columns=('p', 'q'),
            index=('w', 'x', 'y', 'z'),
        )

        x, y = f1.unique_enumerated()
        self.assertEqual(x.tolist(), [[1, 0], [2, 0], [1, 0], [2, 0]])
        self.assertEqual(y.tolist(), [0, 2, 30])

    # ---------------------------------------------------------------------------

    def test_frame_duplicated_a(self) -> None:
        a1 = np.array([[50, 50, 32, 17, 17], [2, 2, 1, 3, 3]])
        f1 = Frame(a1, index=('a', 'b'), columns=('p', 'q', 'r', 's', 't'))

        self.assertEqual(
            f1.duplicated(axis=1).to_pairs(),
            (('p', True), ('q', True), ('r', False), ('s', True), ('t', True)),
        )

        self.assertEqual(f1.duplicated(axis=0).to_pairs(), (('a', False), ('b', False)))

    # ---------------------------------------------------------------------------

    def test_frame_drop_duplicated_a(self) -> None:
        a1 = np.array([[50, 50, 32, 17, 17], [2, 2, 1, 3, 3]])
        f1 = Frame(a1, index=('a', 'b'), columns=('p', 'q', 'r', 's', 't'))

        self.assertEqual(
            f1.drop_duplicated(axis=1, exclude_first=True).to_pairs(axis=1),
            (
                ('a', (('p', 50), ('r', 32), ('s', 17))),
                ('b', (('p', 2), ('r', 1), ('s', 3))),
            ),
        )

    def test_frame_drop_duplicated_b(self) -> None:
        a1 = np.arange(6).reshape((2, 3))
        f1 = Frame(a1, index=('a', 'b'), columns=('p', 'q', 'r'))
        f2 = f1.drop_duplicated(axis=0)
        self.assertEqualFrames(f1, f2)

        with self.assertRaises(NotImplementedError):
            _ = f1.drop_duplicated(axis=-1)

    def test_frame_drop_duplicated_c(self) -> None:
        f1 = Frame.from_records(
            [[1, 2], [1, 2], [3, 3]], index=('a', 'b', 'c'), columns=('p', 'q')
        )
        f2 = f1.drop_duplicated(axis=0)
        self.assertEqual(f2.to_pairs(), (('p', (('c', 3),)), ('q', (('c', 3),))))

    def test_frame_drop_duplicated_d(self) -> None:
        f1 = ff.parse('s(5,5)|v(int, str, float, bool, dtD)')
        f2 = sf.Frame.from_concat((f1, f1, f1), index=sf.IndexAutoFactory)
        f3 = f2.drop_duplicated(exclude_first=True)

        self.assertEqual(f3.shape, (5, 5))
        self.assertEqual([dt.kind for dt in f3.dtypes.values], ['i', 'U', 'f', 'b', 'M'])

    # ---------------------------------------------------------------------------

    def test_frame_from_concat_a(self) -> None:
        records = (
            (2, 2, 'a', False, False),
            (30, 34, 'b', True, False),
        )

        f1 = Frame.from_records(
            records, columns=('p', 'q', 'r', 's', 't'), index=('x', 'a')
        )

        records = (
            (2, 95, 'c', False, False),
            (30, 73, 'd', True, True),
        )

        f2 = Frame.from_records(
            records, columns=('p', 'q', 'r', 's', 't'), index=('x', 'a')
        )

        records = (
            (2, 2, 'a', False, False),
            (30, 73, 'd', True, True),
        )

        f3 = Frame.from_records(
            records, columns=('p', 'q', 'r', 's', 't'), index=('x', 'a')
        )

        f = Frame.from_concat((f1, f2, f3), axis=1, columns=range(15))

        # no blocks are copied or reallcoated
        self.assertEqual(
            f.mloc.tolist(), f1.mloc.tolist() + f2.mloc.tolist() + f3.mloc.tolist()
        )
        # order of index is retained
        self.assertEqual(
            f.to_pairs(axis=1),
            (
                (
                    'x',
                    (
                        (0, 2),
                        (1, 2),
                        (2, 'a'),
                        (3, False),
                        (4, False),
                        (5, 2),
                        (6, 95),
                        (7, 'c'),
                        (8, False),
                        (9, False),
                        (10, 2),
                        (11, 2),
                        (12, 'a'),
                        (13, False),
                        (14, False),
                    ),
                ),
                (
                    'a',
                    (
                        (0, 30),
                        (1, 34),
                        (2, 'b'),
                        (3, True),
                        (4, False),
                        (5, 30),
                        (6, 73),
                        (7, 'd'),
                        (8, True),
                        (9, True),
                        (10, 30),
                        (11, 73),
                        (12, 'd'),
                        (13, True),
                        (14, True),
                    ),
                ),
            ),
        )

    def test_frame_from_concat_b(self) -> None:
        records = (
            (2, 2, 'a', False, False),
            (30, 34, 'b', True, False),
        )

        f1 = Frame.from_records(
            records, columns=('p', 'q', 'r', 's', 't'), index=('x', 'a')
        )

        records = (
            (2, 95, 'c', False, False),
            (30, 73, 'd', True, True),
        )

        f2 = Frame.from_records(
            records, columns=('p', 'q', 'r', 's', 't'), index=('x', 'b')
        )

        records = (
            (2, 2, 'a', False, False),
            (30, 73, 'd', True, True),
        )

        f3 = Frame.from_records(
            records, columns=('p', 'q', 'r', 's', 't'), index=('x', 'c')
        )

        f = Frame.from_concat((f1, f2, f3), axis=1, columns=range(15))

        self.assertEqual(f.index.values.tolist(), ['a', 'b', 'c', 'x'])

        self.assertAlmostEqualFramePairs(
            f.to_pairs(axis=1),
            (
                (
                    'a',
                    (
                        (0, 30),
                        (1, 34),
                        (2, 'b'),
                        (3, True),
                        (4, False),
                        (5, nan),
                        (6, nan),
                        (7, nan),
                        (8, nan),
                        (9, nan),
                        (10, nan),
                        (11, nan),
                        (12, nan),
                        (13, nan),
                        (14, nan),
                    ),
                ),
                (
                    'b',
                    (
                        (0, nan),
                        (1, nan),
                        (2, nan),
                        (3, nan),
                        (4, nan),
                        (5, 30),
                        (6, 73),
                        (7, 'd'),
                        (8, True),
                        (9, True),
                        (10, nan),
                        (11, nan),
                        (12, nan),
                        (13, nan),
                        (14, nan),
                    ),
                ),
                (
                    'c',
                    (
                        (0, nan),
                        (1, nan),
                        (2, nan),
                        (3, nan),
                        (4, nan),
                        (5, nan),
                        (6, nan),
                        (7, nan),
                        (8, nan),
                        (9, nan),
                        (10, 30),
                        (11, 73),
                        (12, 'd'),
                        (13, True),
                        (14, True),
                    ),
                ),
                (
                    'x',
                    (
                        (0, 2),
                        (1, 2),
                        (2, 'a'),
                        (3, False),
                        (4, False),
                        (5, 2),
                        (6, 95),
                        (7, 'c'),
                        (8, False),
                        (9, False),
                        (10, 2),
                        (11, 2),
                        (12, 'a'),
                        (13, False),
                        (14, False),
                    ),
                ),
            ),
        )

        f = Frame.from_concat((f1, f2, f3), union=False, axis=1, columns=range(15))

        self.assertEqual(f.index.values.tolist(), ['x'])
        self.assertEqual(
            f.to_pairs(),
            (
                (0, (('x', 2),)),
                (1, (('x', 2),)),
                (2, (('x', 'a'),)),
                (3, (('x', False),)),
                (4, (('x', False),)),
                (5, (('x', 2),)),
                (6, (('x', 95),)),
                (7, (('x', 'c'),)),
                (8, (('x', False),)),
                (9, (('x', False),)),
                (10, (('x', 2),)),
                (11, (('x', 2),)),
                (12, (('x', 'a'),)),
                (13, (('x', False),)),
                (14, (('x', False),)),
            ),
        )

    def test_frame_from_concat_c(self) -> None:
        records1 = (
            (2, 2, False),
            (30, 34, False),
        )

        f1 = Frame.from_records(records1, columns=('p', 'q', 't'), index=('x', 'a'))

        records2 = (
            ('c', False),
            ('d', True),
        )
        f2 = Frame.from_records(
            records2,
            columns=(
                'r',
                's',
            ),
            index=('x', 'a'),
        )

        # get combined columns as they are unique
        f = Frame.from_concat((f1, f2), axis=1)
        self.assertEqual(
            f.to_pairs(),
            (
                ('p', (('x', 2), ('a', 30))),
                ('q', (('x', 2), ('a', 34))),
                ('t', (('x', False), ('a', False))),
                ('r', (('x', 'c'), ('a', 'd'))),
                ('s', (('x', False), ('a', True))),
            ),
        )

    @skip_win
    def test_frame_from_concat_d(self) -> None:
        records = (
            (2, 2, False),
            (30, 34, False),
        )

        f1 = Frame.from_records(records, columns=('p', 'q', 'r'), index=('a', 'b'))

        records = (
            (2, 2, False),
            (30, 34, False),
        )

        f2 = Frame.from_records(records, columns=('p', 'q', 'r'), index=('c', 'd'))

        f = Frame.from_concat((f1, f2), axis=0)

        # block copmatible will result in attempt to keep vertical types
        self.assertEqual(
            [str(x) for x in f.dtypes.values.tolist()], ['int64', 'int64', 'bool']
        )

        self.assertEqual(
            f.to_pairs(),
            (
                ('p', (('a', 2), ('b', 30), ('c', 2), ('d', 30))),
                ('q', (('a', 2), ('b', 34), ('c', 2), ('d', 34))),
                ('r', (('a', False), ('b', False), ('c', False), ('d', False))),
            ),
        )

    @skip_win
    def test_frame_from_concat_e(self) -> None:
        f1 = Frame.from_items(zip(('a', 'b', 'c'), ((1, 2), (1, 2), (False, True))))

        f = Frame.from_concat((f1, f1, f1), index=range(6))
        self.assertEqual(
            f.to_pairs(),
            (
                ('a', ((0, 1), (1, 2), (2, 1), (3, 2), (4, 1), (5, 2))),
                ('b', ((0, 1), (1, 2), (2, 1), (3, 2), (4, 1), (5, 2))),
                (
                    'c',
                    (
                        (0, False),
                        (1, True),
                        (2, False),
                        (3, True),
                        (4, False),
                        (5, True),
                    ),
                ),
            ),
        )
        self.assertEqual(
            [str(x) for x in f.dtypes.values.tolist()], ['int64', 'int64', 'bool']
        )

        f = Frame.from_concat((f1, f1, f1), axis=1, columns=range(9))

        self.assertEqual(
            f.to_pairs(),
            (
                (0, ((0, 1), (1, 2))),
                (1, ((0, 1), (1, 2))),
                (2, ((0, False), (1, True))),
                (3, ((0, 1), (1, 2))),
                (4, ((0, 1), (1, 2))),
                (5, ((0, False), (1, True))),
                (6, ((0, 1), (1, 2))),
                (7, ((0, 1), (1, 2))),
                (8, ((0, False), (1, True))),
            ),
        )

        self.assertEqual(
            [str(x) for x in f.dtypes.values.tolist()],
            [
                'int64',
                'int64',
                'bool',
                'int64',
                'int64',
                'bool',
                'int64',
                'int64',
                'bool',
            ],
        )

    def test_frame_from_concat_f(self) -> None:
        # force a reblock before concatenating

        a1 = np.array([1, 2, 3], dtype=np.int64)
        a2 = np.array([10, 50, 30], dtype=np.int64)
        a3 = np.array([1345, 2234, 3345], dtype=np.int64)
        a4 = np.array([False, True, False])
        a5 = np.array([False, False, False])
        a6 = np.array(['g', 'd', 'e'])
        tb1 = TypeBlocks.from_blocks((a1, a2, a3, a4, a5, a6))

        f1 = Frame(
            TypeBlocks.from_blocks((a1, a2, a3, a4, a5, a6)),
            columns=('a', 'b', 'c', 'd', 'e', 'f'),
            own_data=True,
        )
        self.assertEqual(len(f1._blocks._blocks), 6)

        f2 = Frame(
            f1.iloc[1:]._blocks.consolidate(),
            columns=('a', 'b', 'c', 'd', 'e', 'f'),
            own_data=True,
        )
        self.assertEqual(len(f2._blocks._blocks), 3)

        f = Frame.from_concat((f1, f2), index=range(5))

        self.assertEqual(
            [str(x) for x in f.dtypes.values.tolist()],
            ['int64', 'int64', 'int64', 'bool', 'bool', '<U1'],
        )

        self.assertEqual(
            [str(x.dtype) for x in f._blocks._blocks], ['int64', 'bool', '<U1']
        )

    def test_frame_from_concat_g(self) -> None:
        records1 = (
            (2, 2, False),
            (30, 34, False),
        )

        f1 = Frame.from_records(records1, columns=('p', 'q', 't'), index=('x', 'a'))

        records2 = (
            ('c', False),
            ('d', True),
        )
        f2 = Frame.from_records(
            records2,
            columns=(
                'r',
                's',
            ),
            index=('x', 'a'),
        )

        # get combined columns as they are unique
        f = Frame.from_concat((f1, f2), axis=1)
        self.assertEqual(
            f.to_pairs(),
            (
                ('p', (('x', 2), ('a', 30))),
                ('q', (('x', 2), ('a', 34))),
                ('t', (('x', False), ('a', False))),
                ('r', (('x', 'c'), ('a', 'd'))),
                ('s', (('x', False), ('a', True))),
            ),
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
        assert post.mloc.tolist() == [f2.mloc[0], f3.mloc[0]]  # type: ignore
        self.assertEqual(post.shape, (2600, 41))

    def test_frame_from_concat_i(self) -> None:
        sf1 = sf.Frame.from_dict(
            dict(a=[1, 2, 3], b=[1, 2, 3]), index=[100, 200, 300]
        ).relabel_level_add(columns='A')
        sf2 = sf.Frame.from_dict(
            dict(a=[1, 2, 3], b=[1, 2, 3]), index=[100, 200, 300]
        ).relabel_level_add(columns='B')

        f = sf.Frame.from_concat((sf1, sf2), axis=1)
        self.assertEqual(
            f.to_pairs(),
            (
                (('A', 'a'), ((100, 1), (200, 2), (300, 3))),
                (('A', 'b'), ((100, 1), (200, 2), (300, 3))),
                (('B', 'a'), ((100, 1), (200, 2), (300, 3))),
                (('B', 'b'), ((100, 1), (200, 2), (300, 3))),
            ),
        )

    def test_frame_from_concat_j(self) -> None:
        sf1 = sf.Frame.from_dict(
            dict(a=[1, 2, 3], b=[1, 2, 3]), index=[100, 200, 300]
        ).relabel_level_add(index='A')
        sf2 = sf.Frame.from_dict(
            dict(a=[1, 2, 3], b=[1, 2, 3]), index=[100, 200, 300]
        ).relabel_level_add(index='B')

        f = sf.Frame.from_concat((sf1, sf2), axis=0)

        self.assertEqual(
            f.to_pairs(),
            (
                (
                    'a',
                    (
                        (('A', 100), 1),
                        (('A', 200), 2),
                        (('A', 300), 3),
                        (('B', 100), 1),
                        (('B', 200), 2),
                        (('B', 300), 3),
                    ),
                ),
                (
                    'b',
                    (
                        (('A', 100), 1),
                        (('A', 200), 2),
                        (('A', 300), 3),
                        (('B', 100), 1),
                        (('B', 200), 2),
                        (('B', 300), 3),
                    ),
                ),
            ),
        )

    def test_frame_from_concat_k(self) -> None:
        records1 = (
            (2, 2, False),
            (30, 34, False),
        )
        f1 = Frame.from_records(records1, columns=('p', 'q', 't'), index=('x', 'a'))

        records2 = (
            ('c', False),
            ('d', True),
        )
        f2 = Frame.from_records(
            records2,
            columns=(
                'r',
                's',
            ),
            index=('x', 'a'),
        )

        # get combined columns as they are unique
        f = Frame.from_concat((f1, f2), axis=1, name='foo')
        self.assertEqual(f.name, 'foo')

    def test_frame_from_concat_m(self) -> None:
        records1 = (
            (2, 2, False),
            (30, 34, False),
        )
        f1 = Frame.from_records(records1, columns=('p', 'q', 't'), index=('x', 'a'))

        records2 = (
            ('c', False),
            ('d', True),
        )
        f2 = Frame.from_records(
            records2,
            columns=(
                3,
                4,
            ),
            index=('x', 'a'),
        )

        f = Frame.from_concat((f1, f2), axis=1, name='foo')

        self.assertEqual(f.columns.values.tolist(), ['p', 'q', 't', 3, 4])
        self.assertEqual(
            f.to_pairs(),
            (
                ('p', (('x', 2), ('a', 30))),
                ('q', (('x', 2), ('a', 34))),
                ('t', (('x', False), ('a', False))),
                (3, (('x', 'c'), ('a', 'd'))),
                (4, (('x', False), ('a', True))),
            ),
        )

    def test_frame_from_concat_n(self) -> None:
        records1 = (
            (2, False),
            (30, False),
        )
        f1 = Frame.from_records(records1, columns=('p', 'q'), index=('x', 'a'))

        records2 = (
            ('c', False),
            ('d', True),
        )
        f2 = Frame.from_records(records2, columns=('p', 'q'), index=(3, 10))

        f = Frame.from_concat((f1, f2), axis=0, name='foo')

        self.assertEqual(f.index.values.tolist(), ['x', 'a', 3, 10])
        self.assertEqual(
            f.to_pairs(),
            (
                ('p', (('x', 2), ('a', 30), (3, 'c'), (10, 'd'))),
                ('q', (('x', False), ('a', False), (3, False), (10, True))),
            ),
        )

    def test_frame_from_concat_o(self) -> None:
        records1 = (
            (2, False),
            (34, False),
        )

        f1 = Frame.from_records(
            records1,
            columns=(
                'p',
                'q',
            ),
            index=('x', 'z'),
        )

        records2 = (
            ('c', False),
            ('d', True),
        )
        f2 = Frame.from_records(
            records2,
            columns=(
                'r',
                's',
            ),
            index=('x', 'z'),
        )

        s1 = Series((0, 100), index=('x', 'z'), name='t')

        f = Frame.from_concat((f1, f2, s1), axis=1)

        self.assertEqual(
            f.to_pairs(),
            (
                ('p', (('x', 2), ('z', 34))),
                ('q', (('x', False), ('z', False))),
                ('r', (('x', 'c'), ('z', 'd'))),
                ('s', (('x', False), ('z', True))),
                ('t', (('x', 0), ('z', 100))),
            ),
        )

    def test_frame_from_concat_p(self) -> None:
        records = (
            (2, False),
            (34, False),
        )
        f1 = Frame.from_records(
            records,
            columns=(
                'p',
                'q',
            ),
            index=('a', 'b'),
        )

        s1 = Series((0, True), index=('p', 'q'), name='c', dtype=object)
        s2 = Series((-2, False), index=('p', 'q'), name='d', dtype=object)

        f = Frame.from_concat((s2, f1, s1), axis=0)

        self.assertEqual(
            f.to_pairs(),
            (
                ('p', (('d', -2), ('a', 2), ('b', 34), ('c', 0))),
                ('q', (('d', False), ('a', False), ('b', False), ('c', True))),
            ),
        )

    def test_frame_from_concat_q(self) -> None:
        s1 = Series(
            (
                2,
                3,
                0,
            ),
            index=list('abc'),
            name='x',
        ).relabel_level_add('i')
        s2 = Series(('10', '20', '100'), index=list('abc'), name='y').relabel_level_add(
            'i'
        )

        # stack horizontally
        f = Frame.from_concat((s1, s2), axis=1)

        self.assertEqual(
            f.to_pairs(),
            (
                ('x', ((('i', 'a'), 2), (('i', 'b'), 3), (('i', 'c'), 0))),
                ('y', ((('i', 'a'), '10'), (('i', 'b'), '20'), (('i', 'c'), '100'))),
            ),
        )

        # stack vertically
        f = Frame.from_concat((s1, s2), axis=0)
        self.assertEqual(
            f.to_pairs(),
            (
                (('i', 'a'), (('x', 2), ('y', '10'))),
                (('i', 'b'), (('x', 3), ('y', '20'))),
                (('i', 'c'), (('x', 0), ('y', '100'))),
            ),
        )

    def test_frame_from_concat_r(self) -> None:
        f1 = sf.Frame.from_dict_records(
            [dict(a=1, b=1), dict(a=2, b=3), dict(a=1, b=1), dict(a=2, b=3)],
            index=sf.IndexHierarchy.from_labels(
                [(1, 'dd'), (1, 'bb'), (2, 'cc'), (2, 'dd')]
            ),
        )

        f2 = (
            sf.Frame.from_dict_records(
                [dict(a=1, b=1), dict(a=2, b=3), dict(a=1, b=1), dict(a=2, b=3)],
                index=sf.IndexHierarchy.from_labels(
                    [(3, 'ddd'), (3, 'bbb'), (4, 'ccc'), (4, 'ddd')]
                ),
            )
            * 100
        )

        self.assertEqual(
            Frame.from_concat((f1, f2), axis=0).to_pairs(),
            (
                (
                    'a',
                    (
                        ((1, 'dd'), 1),
                        ((1, 'bb'), 2),
                        ((2, 'cc'), 1),
                        ((2, 'dd'), 2),
                        ((3, 'ddd'), 100),
                        ((3, 'bbb'), 200),
                        ((4, 'ccc'), 100),
                        ((4, 'ddd'), 200),
                    ),
                ),
                (
                    'b',
                    (
                        ((1, 'dd'), 1),
                        ((1, 'bb'), 3),
                        ((2, 'cc'), 1),
                        ((2, 'dd'), 3),
                        ((3, 'ddd'), 100),
                        ((3, 'bbb'), 300),
                        ((4, 'ccc'), 100),
                        ((4, 'ddd'), 300),
                    ),
                ),
            ),
        )

    def test_frame_from_concat_s(self) -> None:
        records1 = (
            (2, False),
            (34, False),
        )

        f1 = Frame.from_records(
            records1,
            columns=(
                'p',
                'q',
            ),
            index=('x', 'z'),
        )

        records2 = (
            ('c', False),
            ('d', True),
        )
        f2 = Frame.from_records(
            records2,
            columns=(
                'r',
                's',
            ),
            index=('x', 'z'),
        )

        with self.assertRaises(AxisInvalid):
            f = Frame.from_concat((f1, f2), axis=None)

    def test_frame_from_concat_t(self) -> None:
        frame1 = sf.Frame.from_dict_records(
            [dict(a=1, b=1), dict(a=2, b=3), dict(a=1, b=1), dict(a=2, b=3)],
            index=sf.IndexHierarchy.from_labels(
                [(1, 'dd', 0), (1, 'bb', 0), (2, 'cc', 0), (2, 'ee', 0)]
            ),
        )
        frame2 = sf.Frame.from_dict_records(
            [
                dict(a=100, b=200),
                dict(a=20, b=30),
                dict(a=101, b=101),
                dict(a=201, b=301),
            ],
            index=sf.IndexHierarchy.from_labels(
                [(1, 'ddd', 0), (1, 'bbb', 0), (2, 'ccc', 0), (2, 'eee', 0)]
            ),
        )

        frame3 = sf.Frame.from_concat((frame1, frame2))

        self.assertTrue(frame3.columns.equals(frame1.columns))
        self.assertTrue(frame3.columns.equals(frame2.columns))
        self.assertTrue(frame3.index.loc[frame1.index].equals(frame1.index))
        self.assertTrue(frame3.index.loc[frame2.index].equals(frame2.index))

    def test_frame_from_concat_u(self) -> None:
        # this fails; figure out why
        a = sf.Series(('a', 'b', 'c'), index=range(3, 6))
        f = sf.Frame.from_concat(
            (a, sf.Series(a.index.values, index=a.index)),
            axis=0,
            columns=(3, 4, 5),
            index=(1, 2),
        )

        self.assertEqual(
            f.to_pairs(),
            ((3, ((1, 'a'), (2, 3))), (4, ((1, 'b'), (2, 4))), (5, ((1, 'c'), (2, 5)))),
        )

    def test_frame_from_concat_v(self) -> None:
        records1 = (
            (2, False),
            (34, False),
        )

        f1 = Frame.from_records(records1, columns=('p', 'q'), index=('x', 'y'))

        records2 = (
            ('c', False),
            ('d', True),
        )
        f2 = Frame.from_records(
            records2,
            columns=(
                'p',
                'q',
            ),
            index=('x', 'y'),
        )

        # get combined columns as they are unique
        post1 = Frame.from_concat((f1, f2), axis=1, columns=IndexAutoFactory)
        self.assertEqual(
            post1.to_pairs(),
            (
                (0, (('x', 2), ('y', 34))),
                (1, (('x', False), ('y', False))),
                (2, (('x', 'c'), ('y', 'd'))),
                (3, (('x', False), ('y', True))),
            ),
        )

        with self.assertRaises(ErrorInitFrame):
            Frame.from_concat(
                (f1, f2), axis=1, columns=IndexAutoFactory, index=IndexAutoFactory
            )

        post2 = Frame.from_concat((f1, f2), axis=0, index=IndexAutoFactory)
        self.assertEqual(
            post2.to_pairs(),
            (
                ('p', ((0, 2), (1, 34), (2, 'c'), (3, 'd'))),
                ('q', ((0, False), (1, False), (2, False), (3, True))),
            ),
        )

        with self.assertRaises(ErrorInitFrame):
            Frame.from_concat(
                (f1, f2), axis=0, index=IndexAutoFactory, columns=IndexAutoFactory
            )

    @skip_win
    def test_frame_from_concat_w(self) -> None:
        a = sf.Frame.from_dict({0: (1, 2), 1: (2, 3), 2: (True, True)})
        b = sf.Frame.from_dict({0: (1, 2), 1: (np.nan, np.nan), 2: (False, False)})

        # reblock first two columns into integers
        c = a.astype[[0, 1]](int, consolidate_blocks=True)
        self.assertEqual(c._blocks.shapes.tolist(), [(2, 2), (2,)])

        # unaligned blocks compared column to column
        post1 = sf.Frame.from_concat([c, b], index=sf.IndexAutoFactory)

        self.assertEqual(
            post1.dtypes.values.tolist(),
            [np.dtype('int64'), np.dtype('float64'), np.dtype('bool')],
        )

        post2 = sf.Frame.from_concat([a, b], index=sf.IndexAutoFactory)

        self.assertEqual(
            post2.dtypes.values.tolist(),
            [np.dtype('int64'), np.dtype('float64'), np.dtype('bool')],
        )

    def test_frame_from_concat_x(self) -> None:
        f1 = Frame.from_concat([])
        self.assertEqual((0, 0), f1.shape)

        f2 = Frame.from_concat([], columns=('a',))
        self.assertEqual((0, 1), f2.shape)
        self.assertEqual((1,), f2.columns.shape)

        f3 = Frame.from_concat([], index=[])
        self.assertEqual((0, 0), f3.shape)
        self.assertEqual((0,), f3.index.shape)

        f4 = Frame.from_concat([], name='f4')
        self.assertEqual((0, 0), f4.shape)
        self.assertEqual('f4', f4.name)

        f5 = Frame.from_concat([], columns=('a',), index=[], name='f5')
        self.assertEqual((0, 1), f5.shape)
        self.assertEqual((1,), f5.columns.shape)
        self.assertEqual((0,), f5.index.shape)
        self.assertEqual('f5', f5.name)

    def test_frame_from_concat_y1(self) -> None:
        # problematic case of a NaN in IndexHierarchy
        f1 = sf.Frame.from_elements(
            [1, 2], index=IndexHierarchy.from_labels([['b', 'b'], ['b', np.nan]])
        )
        f2 = sf.Frame.from_elements(
            [1, 2], index=IndexHierarchy.from_labels([['b', np.nan], ['b', 'b']])
        )

        f3 = sf.Frame.from_concat((f1, f2), axis=1, columns=['a', 'b'])

        # Since the nans used were singletons, it was able to align the two frames
        self.assertEqual(f3.shape, (2, 2))

        # index order is not stable due to NaN
        self.assertEqual(sorted(f3.values.tolist()), [[1, 2], [2, 1]])
        self.assertEqual(f3.index.depth, 2)
        self.assertAlmostEqualValues(set(f3.index.values.ravel()), {'b', np.nan})

    def test_frame_from_concat_y2(self) -> None:
        # problematic case of a NaN in IndexHierarchy
        f1 = sf.Frame.from_elements(
            [1, 2], index=IndexHierarchy.from_labels([['b', 'b'], ['b', np.nan + 1]])
        )
        f2 = sf.Frame.from_elements(
            [1, 2], index=IndexHierarchy.from_labels([['b', np.nan + 1], ['b', 'b']])
        )

        f3 = sf.Frame.from_concat((f1, f2), axis=1, columns=['a', 'b'])

        # Since the nans used were NOT singletons, it was only able to align part of the frame
        self.assertEqual(f3.shape, (3, 2))

        # index order is not stable due to NaN
        self.assertEqual(
            sorted(f3.fillna(0).values.tolist()), [[0.0, 1.0], [1.0, 2.0], [2.0, 0.0]]
        )
        self.assertEqual(f3.index.depth, 2)

        idx_values = f3.index.values.ravel()
        is_b = idx_values == 'b'
        self.assertTrue(np.any(is_b))
        self.assertTrue(np.isnan(idx_values[~is_b].astype(float)).all())

    def test_frame_from_concat_z(self) -> None:
        frames = tuple(
            sf.Frame.from_element(1, index=sf.Index([i], name='tom'), columns=[str(i)])
            for i in range(2, 4)
        )
        post = Frame.from_concat(frames)
        self.assertEqual(post.index.name, 'tom')
        self.assertEqual(
            post.fillna(None).to_pairs(),
            (('2', ((2, 1.0), (3, None))), ('3', ((2, None), (3, 1.0)))),
        )

    def test_frame_from_concat_aa(self) -> None:
        a1 = np.arange(25).reshape(5, 5)
        a2 = np.arange(start=24, stop=-1, step=-1).reshape(5, 5)

        # Add unique rows/cols
        a1 = np.vstack((np.hstack((a1, np.arange(5).reshape(5, 1))), np.arange(6)))
        a2 = np.vstack((np.hstack((a2, np.arange(5).reshape(5, 1))), np.arange(6)))

        # Determine locations to alter - leave at least one row and col fully unchanged
        to_change = [
            (0, 0),
            (0, 1),
            (0, 3),
            (0, 4),
            (1, 3),
            (3, 1),
            (3, 3),
            (4, 0),
            (4, 3),
        ]
        # Make changes
        for row, col in to_change:
            a1[row][col] = 99

        # Build changes
        f1_col_labels = [
            ['c_I', 'A'],
            ['c_I', 'B'],
            ['c_I', 'C'],
            ['c_II', 'A'],
            ['c_II', 'B'],
            ['c_II', 'C'],
        ]
        f1_idx_labels = [
            ['i_I', '1'],
            ['i_I', '2'],
            ['i_I', '3'],
            ['i_II', '1'],
            ['i_II', '2'],
            ['i_II', '3'],
        ]

        f2_col_labels = [
            ['c_II', 'B'],
            ['c_II', 'A'],
            ['c_I', 'C'],
            ['c_I', 'B'],
            ['c_I', 'A'],
            ['c_I', 'D'],
        ]
        f2_idx_labels = [
            ['i_II', '2'],
            ['i_II', '1'],
            ['i_I', '3'],
            ['i_I', '2'],
            ['i_I', '1'],
            ['i_I', '4'],
        ]

        f1 = sf.Frame(
            a1,
            columns=sf.IndexHierarchy.from_labels(f1_col_labels),
            index=sf.IndexHierarchy.from_labels(f1_idx_labels),
        )
        f2 = sf.Frame(
            a2,
            columns=sf.IndexHierarchy.from_labels(f2_col_labels),
            index=sf.IndexHierarchy.from_labels(f2_idx_labels),
        )
        intersection_cols: sf.Index = f1.columns.intersection(f2.columns)  # type: ignore
        intersection_idx: sf.Index = f1.index.intersection(f2.index)  # type: ignore

        f1_reindexed = f1.reindex(intersection_idx)[intersection_cols]
        f2_reindexed = f2.reindex(intersection_idx)[intersection_cols]

        mismatch_idx_dtypes: sf.Index = f1_reindexed.dtypes != f2_reindexed.dtypes  # type: ignore
        f1_dtypes = f1_reindexed.dtypes[mismatch_idx_dtypes].rename('a')  # type: ignore
        f2_dtypes = f2_reindexed.dtypes[mismatch_idx_dtypes].rename('b')  # type: ignore

        dtype_diffs = sf.Frame.from_concat(
            (f1_dtypes, f2_dtypes),
            axis=1,
            name='dtype_diffs',
        )
        self.assertEqual(dtype_diffs.to_pairs(), (('a', ()), ('b', ())))

    def test_frame_from_concat_bb(self) -> None:
        dt = np.datetime64
        s1 = sf.Series((1, 3), name=dt('2021-01-01'))
        s2 = sf.Series((2, 0), name=dt('1543-10-31'))

        f = Frame.from_concat((s1, s2), index_constructor=IndexDate)
        self.assertIs(f.index.__class__, IndexDate)
        self.assertEqual(
            f.to_pairs(),
            (
                (0, ((dt('2021-01-01'), 1), (dt('1543-10-31'), 2))),
                (1, ((dt('2021-01-01'), 3), (dt('1543-10-31'), 0))),
            ),
        )

    def test_frame_from_concat_cc(self) -> None:
        dt = np.datetime64
        s1 = sf.Series((1, 3), name=dt('2021-01-01'))
        s2 = sf.Series((2, 0), name=dt('1543-10-31'))

        f = Frame.from_concat((s1, s2), axis=1, columns_constructor=IndexDate)
        self.assertIs(f.columns.__class__, IndexDate)
        self.assertEqual(
            f.to_pairs(),
            (
                (dt('2021-01-01'), ((0, 1), (1, 3))),
                (dt('1543-10-31'), ((0, 2), (1, 0))),
            ),
        )

    def test_frame_from_concat_dd(self) -> None:
        a = sf.Series([1], name=np.datetime64('2000-01-01'))
        b = sf.Series([2], name=np.datetime64('2000-01-02'))
        # this catches a complexity of having a name on Series that becomes an index; in this case, since we are passing an index we want to ignore the name on the Series
        f1 = sf.Frame.from_concat((a, b), index=['A', 'B'])
        self.assertEqual(f1.shape, (2, 1))
        self.assertEqual(f1.to_pairs(), ((0, (('A', 1), ('B', 2))),))

    def test_frame_from_concat_ee(self) -> None:
        a = sf.Series([1], name=np.datetime64('2000-01-01'))
        b = sf.Series([2], name=np.datetime64('2000-01-02'))
        f1 = sf.Frame.from_concat((a, b), axis=1, columns_constructor=IndexDate)
        self.assertEqual(f1.shape, (1, 2))
        self.assertIs(f1.columns.__class__, IndexDate)

    def test_frame_from_concat_ff(self) -> None:
        records1 = (
            (2, False),
            (34, False),
        )

        f1 = Frame.from_records(
            records1,
            columns=(
                'p',
                'q',
            ),
            index=('x', 'z'),
        )

        records2 = (
            ('c', False),
            ('d', True),
            ('e', True),
        )
        f2 = Frame.from_records(
            records2,
            columns=(
                'r',
                's',
            ),
            index=('w', 'x', 'z'),
        )

        f3 = Frame.from_concat((f1, f2), axis=1, fill_value=dict(p=-3, q=-2))
        self.assertEqual(
            f3.to_pairs(),
            (
                ('p', (('w', -3), ('x', 2), ('z', 34))),
                ('q', (('w', -2), ('x', False), ('z', False))),
                ('r', (('w', 'c'), ('x', 'd'), ('z', 'e'))),
                ('s', (('w', False), ('x', True), ('z', True))),
            ),
        )

        f4 = Frame.from_concat((f1, f2), axis=1, fill_value=FillValueAuto)
        self.assertEqual(
            f4.to_pairs(),
            (
                ('p', (('w', 0), ('x', 2), ('z', 34))),
                ('q', (('w', False), ('x', False), ('z', False))),
                ('r', (('w', 'c'), ('x', 'd'), ('z', 'e'))),
                ('s', (('w', False), ('x', True), ('z', True))),
            ),
        )

    def test_frame_from_concat_gg(self) -> None:
        # when explicitly named 0, we get the expected error when creting the index
        s1 = Series([1, 2, 3], name=0)
        s2 = s1.rename(np.datetime64('2022-01-01'))
        with self.assertRaises(InvalidDatetime64Initializer):
            _ = Frame.from_concat((s1, s2), axis=1, columns_constructor=sf.IndexDate)

        s1 = Series([1, 2, 3])  # happens implicitly, but here we make it explicit
        s2 = s1.rename(np.datetime64('2022-01-01'))

        with self.assertRaises(InvalidDatetime64Initializer):
            _ = Frame.from_concat((s1, s2), axis=1, columns_constructor=sf.IndexDate)

    def test_frame_from_concat_hh(self) -> None:
        a = sf.Series([1, 2, 3], name='2000-01-01')
        b = sf.Series([4, 5, 6], name=np.datetime64('2000-01-02'))
        # for axis 1, name will become column label, index will remains
        # this fails as we end up using default index constructor on dt64 name, which raises
        f1 = Frame.from_concat((a, b), axis=1, columns_constructor=IndexDate)
        self.assertEqual(
            f1.to_pairs(),
            (
                (np.datetime64('2000-01-01'), ((0, 1), (1, 2), (2, 3))),
                (np.datetime64('2000-01-02'), ((0, 4), (1, 5), (2, 6))),
            ),
        )

        with self.assertRaises(ErrorInitColumns):
            _ = Frame.from_concat((a, b), axis=1, index_constructor=IndexDate)

    # ---------------------------------------------------------------------------

    def test_frame_from_concat_error_init_a(self) -> None:
        f1 = Frame.from_element(
            10,
            columns=(
                'p',
                'q',
            ),
            index=('x', 'z'),
        )
        f2 = Frame.from_element(
            'x',
            columns=(
                'p',
                'q',
            ),
            index=('x', 'z'),
        )

        with self.assertRaises(ErrorInitFrame):
            _ = Frame.from_concat((f1, f2), axis=0)

        with self.assertRaises(ErrorInitFrame):
            _ = Frame.from_concat((f1, f2), axis=1)

        f3 = Frame.from_concat((f1, f2), axis=0, index=IndexAutoFactory)
        self.assertEqual(
            f3.to_pairs(),
            (
                ('p', ((0, 10), (1, 10), (2, 'x'), (3, 'x'))),
                ('q', ((0, 10), (1, 10), (2, 'x'), (3, 'x'))),
            ),
        )

        f4 = Frame.from_concat((f1, f2), axis=1, columns=IndexAutoFactory)
        self.assertEqual(
            f4.to_pairs(),
            (
                (0, (('x', 10), ('z', 10))),
                (1, (('x', 10), ('z', 10))),
                (2, (('x', 'x'), ('z', 'x'))),
                (3, (('x', 'x'), ('z', 'x'))),
            ),
        )

    def test_frame_from_concat_consolidate_blocks_a(self) -> None:
        f1 = Frame.from_element(False, columns=('p', 'q'), index=('x', 'z'))

        f2 = Frame.from_element(
            True,
            columns=(
                'r',
                's',
            ),
            index=('x', 'z'),
        )

        self.assertEqual(
            Frame.from_concat(
                (f1, f2), axis=1, consolidate_blocks=True
            )._blocks.shapes.tolist(),
            [(2, 4)],
        )

    # ---------------------------------------------------------------------------

    def test_frame_from_concat_items_a(self) -> None:
        records1 = (
            (2, 2, False),
            (30, 34, False),
        )

        f1 = Frame.from_records(records1, columns=('p', 'q', 't'), index=('x', 'a'))

        records2 = (
            ('c', False),
            ('d', True),
        )
        f2 = Frame.from_records(
            records2,
            columns=(
                'r',
                's',
            ),
            index=('x', 'a'),
        )

        f3 = Frame.from_concat_items(dict(A=f1, B=f2).items(), axis=1)

        self.assertEqual(
            f3.to_pairs(),
            (
                (('A', 'p'), (('x', 2), ('a', 30))),
                (('A', 'q'), (('x', 2), ('a', 34))),
                (('A', 't'), (('x', False), ('a', False))),
                (('B', 'r'), (('x', 'c'), ('a', 'd'))),
                (('B', 's'), (('x', False), ('a', True))),
            ),
        )

        f4 = FrameGO.from_concat_items(dict(A=f1, B=f2).items(), axis=1)
        self.assertEqual(f4.__class__, FrameGO)
        self.assertEqual(f4.columns.__class__, IndexHierarchyGO)
        self.assertEqual(f4.index.__class__, Index)

        self.assertEqual(
            f4.to_pairs(),
            (
                (('A', 'p'), (('x', 2), ('a', 30))),
                (('A', 'q'), (('x', 2), ('a', 34))),
                (('A', 't'), (('x', False), ('a', False))),
                (('B', 'r'), (('x', 'c'), ('a', 'd'))),
                (('B', 's'), (('x', False), ('a', True))),
            ),
        )

    def test_frame_from_concat_items_b(self) -> None:
        f1 = Frame.from_records(
            ((2, False), (34, False)),
            columns=(
                'p',
                'q',
            ),
            index=('d', 'c'),
        )

        s1 = Series((0, True), index=('p', 'q'), name='c', dtype=object)
        s2 = Series((-2, False), index=('p', 'q'), name='d', dtype=object)

        f2 = Frame.from_concat_items(dict(A=s2, B=f1, C=s1).items(), axis=0)

        self.assertEqual(
            f2.to_pairs(),
            (
                (
                    'p',
                    (
                        (('A', 'd'), -2),
                        (('B', 'd'), 2),
                        (('B', 'c'), 34),
                        (('C', 'c'), 0),
                    ),
                ),
                (
                    'q',
                    (
                        (('A', 'd'), False),
                        (('B', 'd'), False),
                        (('B', 'c'), False),
                        (('C', 'c'), True),
                    ),
                ),
            ),
        )
        self.assertEqual(f2.index.__class__, IndexHierarchy)

    def test_frame_from_concat_items_c(self) -> None:
        f1 = Frame.from_concat_items([])
        self.assertEqual((0, 0), f1.shape)

        f2 = Frame.from_concat_items([], name='f2')
        self.assertEqual((0, 0), f2.shape)
        self.assertEqual('f2', f2.name)

        # Demonstrate the other arguments are inconsequential
        f3 = Frame.from_concat_items(
            [],
            axis=1,
            union=False,
            name='f3',
            fill_value=True,
            consolidate_blocks=False,
        )
        self.assertEqual((0, 0), f3.shape)
        self.assertEqual('f3', f3.name)

    def test_frame_from_concat_items_d(self) -> None:
        s1 = Series((0, True), index=('p', 'q'), name='c', dtype=object)
        s2 = Series((-2, False), index=('p', 'q'), name='d', dtype=object)

        with self.assertRaises(AxisInvalid):
            f1 = Frame.from_concat_items(dict(A=s1, B=s2).items(), axis=2)

    def test_frame_from_concat_items_e(self) -> None:
        records1 = (
            (2, 2, False),
            (30, 34, False),
        )
        f1 = Frame.from_records(records1, columns=('p', 'q', 't'), index=('x', 'a'))

        records2 = (
            ('c', False),
            ('d', True),
        )
        f2 = Frame.from_records(
            records2,
            columns=(
                'r',
                's',
            ),
            index=('x', 'a'),
        )

        # this produces an IH with an outer level of
        f3 = Frame.from_concat_items({('a', 'b'): f1, ('x', 'y'): f2}.items(), axis=1)

        self.assertEqual(f3.shape, (2, 5))
        self.assertEqual(f3.columns.shape, (5, 2))
        self.assertEqual(f3.T.index.shape, (5, 2))

        self.assertEqual(
            f3.to_pairs(),
            (
                ((('a', 'b'), 'p'), (('x', 2), ('a', 30))),
                ((('a', 'b'), 'q'), (('x', 2), ('a', 34))),
                ((('a', 'b'), 't'), (('x', False), ('a', False))),
                ((('x', 'y'), 'r'), (('x', 'c'), ('a', 'd'))),
                ((('x', 'y'), 's'), (('x', False), ('a', True))),
            ),
        )

        self.assertEqual(
            f3.columns.values.tolist(),
            [
                [('a', 'b'), 'p'],
                [('a', 'b'), 'q'],
                [('a', 'b'), 't'],
                [('x', 'y'), 'r'],
                [('x', 'y'), 's'],
            ],
        )

    def test_frame_from_concat_items_f(self) -> None:
        records1 = (
            (2, 2, False),
            (30, 34, False),
        )

        f1 = Frame.from_records(records1, columns=('p', 'q', 't'), index=('x', 'a'))

        records2 = (
            ('c', False),
            ('d', True),
        )

        f2 = Frame.from_records(
            records2,
            columns=(
                'r',
                's',
            ),
            index=('x', 'a'),
        )

        f3 = Frame.from_concat_items(
            dict(A=f1, B=f2).items(), axis=1, columns_constructor=Index
        )
        self.assertEqual(
            f3.to_pairs(),
            (
                (('A', 'p'), (('x', 2), ('a', 30))),
                (('A', 'q'), (('x', 2), ('a', 34))),
                (('A', 't'), (('x', False), ('a', False))),
                (('B', 'r'), (('x', 'c'), ('a', 'd'))),
                (('B', 's'), (('x', False), ('a', True))),
            ),
        )

        f4 = Frame.from_concat_items(
            dict(A=f1, B=f2).items(),
            axis=1,
            columns_constructor=IndexDefaultConstructorFactory('foo'),
        )
        self.assertEqual(f4.columns.name, 'foo')

        with self.assertRaises(NotImplementedError):
            Frame.from_concat_items(
                dict(A=f1, B=f2).items(), axis=1, index_constructor=Index
            )

    def test_frame_from_concat_items_g(self) -> None:
        f1 = Frame.from_records(
            ((2, False), (34, False)),
            columns=(
                'p',
                'q',
            ),
            index=('d', 'c'),
        )

        s1 = Series((0, True), index=('p', 'q'), name='c', dtype=object)
        s2 = Series((-2, False), index=('p', 'q'), name='d', dtype=object)

        f2 = Frame.from_concat_items(
            dict(A=s2, B=f1, C=s1).items(), axis=0, index_constructor=Index
        )
        self.assertEqual(
            f2.to_pairs(),
            (
                (
                    'p',
                    (
                        (('A', 'd'), -2),
                        (('B', 'd'), 2),
                        (('B', 'c'), 34),
                        (('C', 'c'), 0),
                    ),
                ),
                (
                    'q',
                    (
                        (('A', 'd'), False),
                        (('B', 'd'), False),
                        (('B', 'c'), False),
                        (('C', 'c'), True),
                    ),
                ),
            ),
        )

        f3 = Frame.from_concat_items(
            dict(A=s2, B=f1, C=s1).items(),
            axis=0,
            index_constructor=IndexDefaultConstructorFactory('foo'),
        )
        self.assertEqual(f3.index.name, 'foo')

        with self.assertRaises(NotImplementedError):
            Frame.from_concat_items(
                dict(A=s2, B=f1, C=s1).items(), axis=0, columns_constructor=Index
            )

    # ---------------------------------------------------------------------------

    def test_frame_set_index_a(self) -> None:
        records = (
            (2, 2, 'a', False, False),
            (30, 34, 'b', True, False),
        )

        f1 = Frame.from_records(
            records,
            columns=('p', 'q', 'r', 's', 't'),
            index=('x', 'y'),
            consolidate_blocks=True,
        )

        self.assertEqual(
            f1.set_index('r').to_pairs(),
            (
                ('p', (('a', 2), ('b', 30))),
                ('q', (('a', 2), ('b', 34))),
                ('r', (('a', 'a'), ('b', 'b'))),
                ('s', (('a', False), ('b', True))),
                ('t', (('a', False), ('b', False))),
            ),
        )

        self.assertEqual(
            f1.set_index('r', drop=True).to_pairs(),
            (
                ('p', (('a', 2), ('b', 30))),
                ('q', (('a', 2), ('b', 34))),
                ('s', (('a', False), ('b', True))),
                ('t', (('a', False), ('b', False))),
            ),
        )

        f2 = f1.set_index('r', drop=True)

        self.assertEqual(
            f2.to_pairs(),
            (
                ('p', (('a', 2), ('b', 30))),
                ('q', (('a', 2), ('b', 34))),
                ('s', (('a', False), ('b', True))),
                ('t', (('a', False), ('b', False))),
            ),
        )

        self.assertTrue(f1.mloc[[0, 2]].tolist() == f2.mloc.tolist())

    def test_frame_set_index_b(self) -> None:
        records = (
            (2, 2, 'a', False, True),
            (30, 34, 'b', True, False),
        )
        f1 = Frame.from_records(
            records,
            columns=('p', 'q', 'r', 's', 't'),
            index=('x', 'y'),
            consolidate_blocks=True,
        )

        for col in f1.columns:
            f2 = f1.set_index(col)
            self.assertEqual(f2.index.name, col)

    def test_frame_set_index_c(self) -> None:
        records = (
            (2, 2, 'a', False, True),
            (30, 34, 'b', True, False),
        )
        f1 = Frame.from_records(
            records,
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

    def test_frame_set_index_e(self) -> None:
        f1 = ff.parse('s(3,5)|v(str)|i(I, int)')

        f2 = f1.set_index([1, 2])
        self.assertEqual(f2.index.name, (1, 2))
        self.assertEqual(
            f2.to_pairs(),
            (
                (
                    0,
                    (
                        (('zaji', 'ztsv'), 'zjZQ'),
                        (('zJnC', 'zUvW'), 'zO5l'),
                        (('zDdR', 'zkuW'), 'zEdH'),
                    ),
                ),
                (
                    1,
                    (
                        (('zaji', 'ztsv'), 'zaji'),
                        (('zJnC', 'zUvW'), 'zJnC'),
                        (('zDdR', 'zkuW'), 'zDdR'),
                    ),
                ),
                (
                    2,
                    (
                        (('zaji', 'ztsv'), 'ztsv'),
                        (('zJnC', 'zUvW'), 'zUvW'),
                        (('zDdR', 'zkuW'), 'zkuW'),
                    ),
                ),
                (
                    3,
                    (
                        (('zaji', 'ztsv'), 'z2Oo'),
                        (('zJnC', 'zUvW'), 'z5l6'),
                        (('zDdR', 'zkuW'), 'zCE3'),
                    ),
                ),
                (
                    4,
                    (
                        (('zaji', 'ztsv'), 'zDVQ'),
                        (('zJnC', 'zUvW'), 'z5hI'),
                        (('zDdR', 'zkuW'), 'zyT8'),
                    ),
                ),
            ),
        )
        f3 = f1.set_index(slice(2, None), drop=True)
        self.assertEqual(f3.index.name, (2, 3, 4))
        self.assertEqual(
            f3.to_pairs(),
            (
                (
                    0,
                    (
                        (('ztsv', 'z2Oo', 'zDVQ'), 'zjZQ'),
                        (('zUvW', 'z5l6', 'z5hI'), 'zO5l'),
                        (('zkuW', 'zCE3', 'zyT8'), 'zEdH'),
                    ),
                ),
                (
                    1,
                    (
                        (('ztsv', 'z2Oo', 'zDVQ'), 'zaji'),
                        (('zUvW', 'z5l6', 'z5hI'), 'zJnC'),
                        (('zkuW', 'zCE3', 'zyT8'), 'zDdR'),
                    ),
                ),
            ),
        )

    def test_frame_set_index_f(self) -> None:
        f1 = Frame.from_records([(1, 2, 3)])
        f2 = f1.set_index(None, drop=True)
        self.assertTrue(f1.equals(f2))

    def test_frame_set_index_g(self) -> None:
        dt64 = np.datetime64
        f1 = ff.parse('s(2,4)|v(dtD, dtY, int, str)')
        f2 = f1.set_index([0, 1])
        self.assertEqual(f2.index[0], (dt64('2210-12-26'), dt64('164167')))

    # ---------------------------------------------------------------------------

    def test_frame_set_columns_a1(self) -> None:
        records = (
            (2, 2, 'a', False, False),
            (30, 34, 'b', True, False),
            (9, 4, 'c', False, True),
        )

        f1 = Frame.from_records(records, index=('a', 'b', 'c'))
        f2 = f1.set_columns('c')
        self.assertEqual(f2.columns.name, 'c')
        self.assertEqual(
            f2.to_pairs(),
            (
                (9, (('a', 2), ('b', 30), ('c', 9))),
                (4, (('a', 2), ('b', 34), ('c', 4))),
                ('c', (('a', 'a'), ('b', 'b'), ('c', 'c'))),
                (False, (('a', False), ('b', True), ('c', False))),
                (True, (('a', False), ('b', False), ('c', True))),
            ),
        )

        f3 = f1.set_columns('c', drop=True)
        self.assertEqual(f3.columns.name, 'c')
        self.assertEqual(
            f3.to_pairs(),
            (
                (9, (('a', 2), ('b', 30))),
                (4, (('a', 2), ('b', 34))),
                ('c', (('a', 'a'), ('b', 'b'))),
                (False, (('a', False), ('b', True))),
                (True, (('a', False), ('b', False))),
            ),
        )

    def test_frame_set_columns_a2(self) -> None:
        records = (
            (2, 2, 'a', False, False),
            (30, 34, 'b', True, False),
            (9, 4, 'c', False, True),
        )

        f1 = Frame.from_records(records, index=('a', 'b', 'c'))
        f2 = f1.set_columns(
            ['a', 'c'],
        )
        self.assertEqual(f2.columns.name, ('a', 'c'))
        self.assertEqual(
            f2.to_pairs(),
            (
                ((2, 9), (('a', 2), ('b', 30), ('c', 9))),
                ((2, 4), (('a', 2), ('b', 34), ('c', 4))),
                (('a', 'c'), (('a', 'a'), ('b', 'b'), ('c', 'c'))),
                ((False, False), (('a', False), ('b', True), ('c', False))),
                ((False, True), (('a', False), ('b', False), ('c', True))),
            ),
        )

        f3 = f1.set_columns(['a', 'c'], drop=True)
        self.assertEqual(f3.columns.name, ('a', 'c'))
        self.assertEqual(
            f3.to_pairs(),
            (
                ((2, 9), (('b', 30),)),
                ((2, 4), (('b', 34),)),
                (('a', 'c'), (('b', 'b'),)),
                ((False, False), (('b', True),)),
                ((False, True), (('b', False),)),
            ),
        )

    def test_frame_set_columns_a3(self) -> None:
        records = (
            (2, 2, 'a', False, False),
            (30, 34, 'b', True, False),
            (9, 4, 'c', False, True),
        )
        f1 = Frame.from_records(records, index=('a', 'b', 'c'))
        f2 = f1.set_columns([])
        self.assertTrue(f1.equals(f2))

    # ---------------------------------------------------------------------------
    def test_frame_set_columns_hierarchy_a1(self) -> None:
        records = (
            (2, 2, 'a', False, False),
            (30, 34, 'b', True, False),
            (9, 4, 'c', False, True),
        )

        f1 = Frame.from_records(records, index=('a', 'b', 'c'))
        f2 = f1.set_columns_hierarchy(
            ['a', 'c'],
        )
        self.assertEqual(f2.columns.name, ('a', 'c'))
        self.assertEqual(f2.columns.depth, 2)
        self.assertEqual(
            f2.to_pairs(),
            (
                ((2, 9), (('a', 2), ('b', 30), ('c', 9))),
                ((2, 4), (('a', 2), ('b', 34), ('c', 4))),
                (('a', 'c'), (('a', 'a'), ('b', 'b'), ('c', 'c'))),
                ((False, False), (('a', False), ('b', True), ('c', False))),
                ((False, True), (('a', False), ('b', False), ('c', True))),
            ),
        )

        f3 = f1.set_columns_hierarchy(['a', 'c'], drop=True)
        self.assertEqual(f3.columns.name, ('a', 'c'))
        self.assertEqual(f3.columns.depth, 2)
        self.assertEqual(
            f3.to_pairs(),
            (
                ((2, 9), (('b', 30),)),
                ((2, 4), (('b', 34),)),
                (('a', 'c'), (('b', 'b'),)),
                ((False, False), (('b', True),)),
                ((False, True), (('b', False),)),
            ),
        )

    def test_frame_set_columns_hierarchy_a2(self) -> None:
        records = (
            (2, 2, 'a', False, False),
            (30, 34, 'b', True, False),
            (9, 4, 'c', False, True),
        )

        f1 = Frame.from_records(records, index=('a', 'b', 'c'))
        f2 = f1.set_columns_hierarchy(
            ('a', 'c'),
        )  # this treated as a non-element selection
        self.assertEqual(f2.columns.depth, 2)

    def test_frame_set_columns_hierarchy_b(self) -> None:
        records = (
            (1, 2, 1, 2),
            (10, 10, 20, 20),
            (0, 1, 2, 3),
        )
        f1 = Frame.from_records(records, index=('a', 'b', 'c'))
        f2 = f1.set_columns_hierarchy(['a', 'b'], reorder_for_hierarchy=True, drop=True)
        self.assertEqual(
            f2.to_pairs(),
            (
                ((1, 10), (('c', 0),)),
                ((1, 20), (('c', 2),)),
                ((2, 10), (('c', 1),)),
                ((2, 20), (('c', 3),)),
            ),
        )
        self.assertEqual(f2.columns.depth, 2)

    # ---------------------------------------------------------------------------

    def test_frame_unset_columns_a1(self) -> None:
        records = (
            (2, 2, 'a', False),
            (30, 34, 'b', True),
        )
        f1 = Frame.from_records(
            records,
            index=('a', 'b'),
            consolidate_blocks=True,
            columns=IndexHierarchy.from_product((10, 20), (1, 2)),
        )
        f2 = f1.unset_columns()
        self.assertEqual(
            f2.to_pairs(),
            (
                (0, (('__index0__', 10), ('__index1__', 1), ('a', 2), ('b', 30))),
                (1, (('__index0__', 10), ('__index1__', 2), ('a', 2), ('b', 34))),
                (2, (('__index0__', 20), ('__index1__', 1), ('a', 'a'), ('b', 'b'))),
                (3, (('__index0__', 20), ('__index1__', 2), ('a', False), ('b', True))),
            ),
        )

    def test_frame_unset_columns_a2(self) -> None:
        records = (
            (2, 2, 'a', False),
            (30, 34, 'b', True),
        )
        f1 = Frame.from_records(
            records,
            index=IndexHierarchy.from_labels((('A', 'a'), ('A', 'b'))),
            consolidate_blocks=True,
            columns=IndexHierarchy.from_product((10, 20), (1, 2)),
        )
        with self.assertRaises(RuntimeError):
            _ = f1.unset_columns()

    def test_frame_unset_columns_a3(self) -> None:
        records = (
            (2, 2, 'a'),
            (30, 34, 'b'),
        )
        f1 = FrameGO.from_records(records)
        f1['x'] = False

        f2 = f1.unset_columns()
        self.assertEqual(
            f2.to_pairs(),
            (
                (0, (('__index0__', 0), (0, 2), (1, 30))),
                (1, (('__index0__', 1), (0, 2), (1, 34))),
                (2, (('__index0__', 2), (0, 'a'), (1, 'b'))),
                (3, (('__index0__', 'x'), (0, False), (1, False))),
            ),
        )

    def test_frame_unset_columns_a4(self) -> None:
        records = (
            (2, 2, 'a', False),
            (30, 34, 'b', True),
        )
        f1 = Frame.from_records(
            records,
            index=IndexHierarchy.from_labels((('A', 'a'), ('A', 'b'))),
            consolidate_blocks=True,
            columns=('a', 'b', 'c', 'd'),
        )

        with self.assertRaises(RuntimeError):
            _ = f1.unset_columns()

        f2 = f1.unset_columns([('B', 'b')])
        self.assertEqual(f2.index.values.tolist(), [['B', 'b'], ['A', 'a'], ['A', 'b']])

    def test_frame_unset_columns_a5(self) -> None:
        records = (
            (2, 2, 'a', False),
            (30, 34, 'b', True),
        )
        f1 = Frame.from_records(
            records,
            index=IndexHierarchy.from_labels((('A', 'a'), ('A', 'b'))),
            consolidate_blocks=True,
            columns=IndexHierarchy.from_product((10, 20), (1, 2)),
        )

        with self.assertRaises(RuntimeError):
            _ = f1.unset_columns()

        with self.assertRaises(RuntimeError):
            _ = f1.unset_columns(('a',))

        f2 = f1.unset_columns((('B', 'a'), ('B', 'c')))
        self.assertEqual(
            f2.to_pairs(),
            (
                (
                    0,
                    (
                        (('B', 'a'), 10),
                        (('B', 'c'), 1),
                        (('A', 'a'), 2),
                        (('A', 'b'), 30),
                    ),
                ),
                (
                    1,
                    (
                        (('B', 'a'), 10),
                        (('B', 'c'), 2),
                        (('A', 'a'), 2),
                        (('A', 'b'), 34),
                    ),
                ),
                (
                    2,
                    (
                        (('B', 'a'), 20),
                        (('B', 'c'), 1),
                        (('A', 'a'), 'a'),
                        (('A', 'b'), 'b'),
                    ),
                ),
                (
                    3,
                    (
                        (('B', 'a'), 20),
                        (('B', 'c'), 2),
                        (('A', 'a'), False),
                        (('A', 'b'), True),
                    ),
                ),
            ),
        )

    def test_frame_unset_columns_b1(self) -> None:
        records = (
            (2, False),
            (3, True),
            (4, False),
        )
        f1 = sf.Frame.from_records(
            records, index=('idx1', 'idx2', 'idx3'), columns=('col1', 'col2')
        )
        f2 = f1.unset_columns(drop=True)

        self.assertEqual(
            f2.to_pairs(),
            (
                (0, (('idx1', 2), ('idx2', 3), ('idx3', 4))),
                (1, (('idx1', False), ('idx2', True), ('idx3', False))),
            ),
        )
        self.assertEqual(f2.columns.values.tolist(), [0, 1])

    def test_frame_unset_columns_b2(self) -> None:
        records = (
            (1, 'a', 10, 100),
            (2, 'b', 20, 200),
            (3, 'c', 30, 300),
        )
        columns = sf.IndexHierarchy.from_product(('A', 'B'), ('x', 'y'))
        f1 = sf.Frame.from_records(
            records, index=('idx1', 'idx2', 'idx3'), columns=columns
        )
        f2 = f1.unset_columns(drop=True)

        self.assertEqual(
            f2.to_pairs(),
            (
                (0, (('idx1', 1), ('idx2', 2), ('idx3', 3))),
                (1, (('idx1', 'a'), ('idx2', 'b'), ('idx3', 'c'))),
                (2, (('idx1', 10), ('idx2', 20), ('idx3', 30))),
                (3, (('idx1', 100), ('idx2', 200), ('idx3', 300))),
            ),
        )
        self.assertEqual(f2.columns.values.tolist(), [0, 1, 2, 3])

    def test_frame_unset_columns_b3(self) -> None:
        records = ((2, 2), (30, 3), (2, -95))
        f1 = sf.Frame.from_records(records, columns=('a', 'b'))
        with self.assertRaises(RuntimeError):
            f1.unset_columns(('unset_col',), drop=True)

    def test_frame_unset_columns_b4(self) -> None:
        records = ((1, 'a'), (2, 'b'), (3, 'c'))
        f_go = FrameGO.from_records(
            records, columns=('col1', 'col2'), index=('x', 'y', 'z')
        )
        f_immutable = f_go.unset_columns(drop=True)
        f_go['col3'] = [True, False, False]
        # Ensure Frame does not get mutated after call and growing FrameGO
        self.assertEqual(
            f_immutable.to_pairs(),
            (
                (0, (('x', 1), ('y', 2), ('z', 3))),
                (1, (('x', 'a'), ('y', 'b'), ('z', 'c'))),
            ),
        )

    # ---------------------------------------------------------------------------

    def test_frame_head_tail_a(self) -> None:
        # thest of multi threaded apply

        f1 = Frame.from_items(zip(range(10), (np.random.rand(1000) for _ in range(10))))
        self.assertEqual(f1.head(3).index.values.tolist(), [0, 1, 2])
        self.assertEqual(f1.tail(3).index.values.tolist(), [997, 998, 999])

    # ---------------------------------------------------------------------------

    def test_frame_from_records_date_a(self) -> None:
        d = np.datetime64

        records = (
            (d('2018-01-02'), d('2018-01-02'), 'a', False, False),
            (d('2017-01-02'), d('2017-01-02'), 'b', True, False),
            (d('2016-01-02'), d('2016-01-02'), 'c', False, False),
            (d('2015-01-02'), d('2015-01-02'), 'd', True, True),
        )

        f1 = Frame.from_records(records, columns=('p', 'q', 'r', 's', 't'), index=None)

        dtype = np.dtype

        self.assertEqual(
            list(f1._blocks._reblock_signature()),
            [(dtype('<M8[D]'), 2), (dtype('<U1'), 1), (dtype('bool'), 2)],
        )

    # ---------------------------------------------------------------------------

    def test_frame_from_records_a(self) -> None:
        NT = namedtuple('NT', ('a', 'b', 'c'))
        records = [NT(x, x, x) for x in range(4)]
        f1 = Frame.from_records(records)
        self.assertEqual(f1.columns.values.tolist(), ['a', 'b', 'c'])
        self.assertEqual(f1.sum().to_pairs(), (('a', 6), ('b', 6), ('c', 6)))

    def test_frame_from_records_b(self) -> None:
        f1 = sf.Frame.from_records([[1, 2], [2, 3]], columns=['a', 'b'])
        self.assertEqual(
            f1.to_pairs(), (('a', ((0, 1), (1, 2))), ('b', ((0, 2), (1, 3))))
        )

        with self.assertRaises(Exception):
            f2 = sf.Frame.from_records([[1, 2], [2, 3]], columns=['a'])

    def test_frame_from_records_c(self) -> None:
        s1 = Series([3, 4, 5], index=('x', 'y', 'z'))
        s2 = Series(list('xyz'), index=('x', 'y', 'z'))

        with self.assertRaises(Exception):
            # cannot use Series in from_records
            f1 = sf.Frame.from_records([s1, s2], columns=['a', 'b', 'c'])

    def test_frame_from_records_d(self) -> None:
        a1 = np.array([[1, 2, 3], [4, 5, 6]])

        f1 = sf.Frame.from_records(
            a1, index=('x', 'y'), columns=['a', 'b', 'c'], name='foo'
        )

        self.assertEqual(
            f1.to_pairs(),
            (
                ('a', (('x', 1), ('y', 4))),
                ('b', (('x', 2), ('y', 5))),
                ('c', (('x', 3), ('y', 6))),
            ),
        )
        self.assertEqual(f1.name, 'foo')

    def test_frame_from_records_e(self) -> None:
        records = [[1, '2', 3], [4, '5', 6]]
        dtypes = (np.int64, str, str)
        f1 = sf.Frame.from_records(
            records, index=('x', 'y'), columns=['a', 'b', 'c'], dtypes=dtypes
        )
        self.assertEqual(
            f1.dtypes.iter_element().apply(str).to_pairs(),
            (('a', 'int64'), ('b', '<U1'), ('c', '<U1')),
        )

    def test_frame_from_records_f(self) -> None:
        records = [[1, '2', 3], [4, '5', 6]]
        dtypes = {'b': np.int64}
        f1 = sf.Frame.from_records(
            records, index=('x', 'y'), columns=['a', 'b', 'c'], dtypes=dtypes
        )

        self.assertEqual(str(f1.dtypes['b']), 'int64')

    def test_frame_from_records_g(self) -> None:
        NT = namedtuple('NT', ('a', 'b', 'c'))

        records = [NT(1, '2', 3), NT(4, '5', 6)]
        dtypes = {'b': np.int64}
        f1 = sf.Frame.from_records(records, dtypes=dtypes)

        self.assertEqual(str(f1.dtypes['b']), 'int64')

    def test_frame_from_records_h(self) -> None:
        with self.assertRaises(ErrorInitFrame):
            Frame.from_records(())

        with self.assertRaises(ErrorInitFrame):
            Frame.from_records(((0, 1, 2) for x in range(3) if x < 0))

    def test_frame_from_records_i(self) -> None:
        f1 = sf.Frame.from_records([(88,), (27,), (27,), (None,)])

        self.assertEqual(f1.to_pairs(), ((0, ((0, 88), (1, 27), (2, 27), (3, None))),))

    def test_frame_from_records_j(self) -> None:
        records = [dict(a=1, b=2), dict(b=10, c=4), dict(c=20, d=-1)]
        with self.assertRaises(ErrorInitFrame):
            # cannot supply columns when records are dictionaries
            f1 = Frame.from_records(records, columns=('b', 'c', 'd'))

    def test_frame_from_records_k(self) -> None:
        def gen() -> tp.Iterator[int]:
            empty: tp.Iterable[int] = ()
            for k in empty:
                yield k

        f1 = Frame.from_records(gen(), columns=('a', 'b', 'c'))
        self.assertEqual(f1.to_pairs(), (('a', ()), ('b', ()), ('c', ())))

    def test_frame_from_records_m(self) -> None:
        records = np.arange(4).reshape((2, 2))
        dtypes = (bool, bool)
        with self.assertRaises(ErrorInitFrame):
            f1 = sf.Frame.from_records(records, dtypes=dtypes)

    def test_frame_from_records_n(self) -> None:
        mapping = {10: (3, 4), 50: (5, 6)}
        f1 = sf.Frame.from_records(mapping.values())
        self.assertEqual(f1.to_pairs(), ((0, ((0, 3), (1, 5))), (1, ((0, 4), (1, 6)))))

    def test_frame_from_records_o(self) -> None:
        records = (('x', 't'), (1, 2))
        f1 = sf.Frame.from_records(records)

        self.assertEqual(
            f1.to_pairs(), ((0, ((0, 'x'), (1, 1))), (1, ((0, 't'), (1, 2))))
        )

    def test_frame_from_records_p(self) -> None:
        records = [('x', 't')] * 10 + [(1, 2)]
        f1 = sf.Frame.from_records(records)
        self.assertEqual(
            f1.to_pairs(),
            (
                (
                    0,
                    (
                        (0, 'x'),
                        (1, 'x'),
                        (2, 'x'),
                        (3, 'x'),
                        (4, 'x'),
                        (5, 'x'),
                        (6, 'x'),
                        (7, 'x'),
                        (8, 'x'),
                        (9, 'x'),
                        (10, 1),
                    ),
                ),
                (
                    1,
                    (
                        (0, 't'),
                        (1, 't'),
                        (2, 't'),
                        (3, 't'),
                        (4, 't'),
                        (5, 't'),
                        (6, 't'),
                        (7, 't'),
                        (8, 't'),
                        (9, 't'),
                        (10, 2),
                    ),
                ),
            ),
        )

    def test_frame_from_records_q(self) -> None:
        # Y: tp.Type[tp.NamedTuple]

        class Y(tp.NamedTuple):
            x: str
            y: int

        f0 = Frame.from_records([(Y('foo', 1), 1, 2)])
        f1 = Frame.from_records([(1, 2, Y('foo', 1))])
        f2 = Frame.from_records([(1, 2, ('foo', 1))])

        self.assertEqual(f0.shape, f1.shape)
        self.assertEqual(f0.shape, f2.shape)

        self.assertEqual(
            f0.to_pairs(),
            ((0, ((0, Y(x='foo', y=1)),)), (1, ((0, 1),)), (2, ((0, 2),))),
        )
        self.assertEqual(
            f1.to_pairs(),
            ((0, ((0, 1),)), (1, ((0, 2),)), (2, ((0, Y(x='foo', y=1)),))),
        )
        self.assertEqual(
            f2.to_pairs(), ((0, ((0, 1),)), (1, ((0, 2),)), (2, ((0, ('foo', 1)),)))
        )

    def test_frame_from_records_r(self) -> None:
        @dataclasses.dataclass
        class Item:
            left: str
            right: int

        records = (Item('a', 3), Item('b', 20), Item('c', -34))
        f1 = Frame.from_records(records, index=tuple('xyz'))
        self.assertEqual(
            f1.to_pairs(),
            (
                ('left', (('x', 'a'), ('y', 'b'), ('z', 'c'))),
                ('right', (('x', 3), ('y', 20), ('z', -34))),
            ),
        )
        f2 = Frame.from_records_items(zip(tuple('xyz'), records))
        self.assertEqual(
            f2.to_pairs(),
            (
                ('left', (('x', 'a'), ('y', 'b'), ('z', 'c'))),
                ('right', (('x', 3), ('y', 20), ('z', -34))),
            ),
        )

    def test_frame_from_records_s(self) -> None:
        records = ((10, 20), (0, 2), (5, 399))
        f1 = sf.Frame.from_records(records, dtypes=str)
        self.assertEqual(f1.values.tolist(), [['10', '20'], ['0', '2'], ['5', '399']])

        f2 = sf.Frame.from_records(records, dtypes=bool)
        self.assertEqual(f2.values.tolist(), [[True, True], [False, True], [True, True]])

    def test_frame_from_records_t(self) -> None:
        kv = {'x': int, 'y': str}
        f1 = sf.Frame.from_records(
            [('10', 5), ('3', 20)], columns=kv.keys(), dtypes=kv.values()
        )
        self.assertEqual(
            f1.to_pairs(), (('x', ((0, 10), (1, 3))), ('y', ((0, '5'), (1, '20'))))
        )

    def test_frame_from_recrods_u(self) -> None:
        f1 = Frame.from_element(1, index=[1, 2], columns=['a', 'b'])
        f2 = Frame.from_records(
            f1.iter_tuple(constructor=tuple), columns=f1.columns, dtypes=f1.dtypes
        )
        self.assertEqual(
            f2.to_pairs(), (('a', ((0, 1), (1, 1))), ('b', ((0, 1), (1, 1))))
        )

    def test_frame_from_recrods_v(self) -> None:
        with self.assertRaises(NotImplementedError):
            f1 = Frame.from_records(
                ((x for x in range(3)), (x for x in range(3))),
                index=[1, 2],
                columns=['a', 'b', 'c'],
            )

    def test_frame_from_records_w(self) -> None:
        records = [[1, True], [4, False]]
        dtypes = {'b': int, 'c': int}  # NOTE: can overspecify
        f1 = sf.Frame.from_records(
            records, index=('x', 'y'), columns=['a', 'b'], dtypes=dtypes
        )
        self.assertEqual(
            [d.kind for d in f1.dtypes.values],
            ['i', 'i'],
        )

    def test_frame_from_records_x(self) -> None:
        f1 = sf.Frame.from_records((), columns=list('ABC'), dtypes=str)
        self.assertEqual(f1.dtypes.unique().tolist(), [np.dtype('U1')])

    # ---------------------------------------------------------------------------

    def test_frame_from_dict_records_a(self) -> None:
        records = [{'a': x, 'b': x, 'c': x} for x in range(4)]
        f1 = Frame.from_dict_records(records)
        self.assertEqual(f1.columns.values.tolist(), ['a', 'b', 'c'])
        self.assertEqual(f1.sum().to_pairs(), (('a', 6), ('b', 6), ('c', 6)))

    def test_frame_from_dict_records_b(self) -> None:
        # handle case of dict views
        a = {
            1: {
                'a': 1,
                'b': 2,
            },
            2: {
                'a': 4,
                'b': 3,
            },
        }

        post = Frame.from_dict_records(a.values(), index=list(a.keys()))

        self.assertEqual(
            post.to_pairs(), (('a', ((1, 1), (2, 4))), ('b', ((1, 2), (2, 3))))
        )

    def test_frame_from_dict_records_c(self) -> None:
        records = [dict(a=1, b='2', c=3), dict(a=4, b='5', c=6)]
        dtypes = {'b': np.int64}
        f1 = sf.Frame.from_dict_records(records, dtypes=dtypes)

        self.assertEqual(str(f1.dtypes['b']), 'int64')

    def test_frame_from_dict_records_d(self) -> None:
        records = [dict(a=1, b=2), dict(b=10, c=4), dict(c=20, d=-1)]
        f1 = Frame.from_dict_records(records, fill_value=0)
        self.assertEqual(
            f1.to_pairs(),
            (
                ('a', ((0, 1), (1, 0), (2, 0))),
                ('b', ((0, 2), (1, 10), (2, 0))),
                ('c', ((0, 0), (1, 4), (2, 20))),
                ('d', ((0, 0), (1, 0), (2, -1))),
            ),
        )

    def test_frame_from_dict_records_e(self) -> None:
        # handle case of dict views
        a = {
            1: {
                'a': 1,
                'b': 2,
            },
            2: {
                'a': 4,
                'b': 3,
            },
        }

        post = Frame.from_dict_records(
            a.values(), index=list(a.keys()), dtypes=(int, str)
        )
        self.assertEqual(
            post.to_pairs(), (('a', ((1, 1), (2, 4))), ('b', ((1, '2'), (2, '3'))))
        )

    def test_frame_from_dict_records_f(self) -> None:
        # handle case of dict views
        data = ()
        with self.assertRaises(ErrorInitFrame):
            _ = Frame.from_dict_records(data)

    def test_frame_from_dict_records_g(self) -> None:
        records = [
            dict(a=True, b=False),
            dict(a=True, b=False),
        ]
        f1 = Frame.from_dict_records(records, consolidate_blocks=True)
        self.assertEqual(f1._blocks.shapes.tolist(), [(2, 2)])

    def test_frame_from_dict_records_h(self) -> None:
        records = [
            dict(a=True, b=False),
            dict(b=True, c=False),
        ]
        f1 = Frame.from_dict_records(records, fill_value='x')
        self.assertEqual(
            f1.to_pairs(),
            (
                ('a', ((0, True), (1, 'x'))),
                ('b', ((0, False), (1, True))),
                ('c', ((0, 'x'), (1, False))),
            ),
        )

    def test_frame_from_dict_records_i(self) -> None:
        records = [dict(a=1, b='2', c=3), dict(b='5')]
        dtypes = None  # {'b': np.int64}
        f1 = sf.Frame.from_dict_records(
            records, dtypes=dtypes, fill_value=dict(a=-1, c=-3)
        )
        self.assertEqual(
            f1.to_pairs(),
            (
                ('a', ((0, 1), (1, -1))),
                ('b', ((0, '2'), (1, '5'))),
                ('c', ((0, 3), (1, -3))),
            ),
        )

    def test_frame_from_dict_records_j(self) -> None:
        records = [dict(a=1, b='2', c=3), dict(c='5')]
        dtypes = [bool, str, int]
        f1 = sf.Frame.from_dict_records(records, dtypes=dtypes, fill_value=FillValueAuto)
        self.assertEqual(
            f1.to_pairs(),
            (
                ('a', ((0, True), (1, False))),
                ('b', ((0, '2'), (1, ''))),
                ('c', ((0, 3), (1, 5))),
            ),
        )

    # ---------------------------------------------------------------------------

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

        f1 = Frame.from_json_records(msg, name=msg)
        self.assertEqual(
            sorted(f1.columns.values.tolist()),
            sorted(['completed', 'id', 'title', 'userId']),
        )
        self.assertEqual(f1['id'].sum(), 10)

        self.assertEqual(f1.name, msg)

    # ---------------------------------------------------------------------------

    def test_frame_relabel_flat_a(self) -> None:
        records = (
            (1, 2, 'a', False, True),
            (30, 34, 'b', True, False),
            (54, 95, 'c', False, False),
        )

        columns = IndexHierarchy.from_labels(
            (('a', 1), ('a', 2), ('b', 1), ('b', 2), ('b', 3))
        )
        f1 = Frame.from_records(records, columns=columns, index=('x', 'y', 'z'))

        f2 = f1.relabel_flat(columns=True)

        self.assertEqual(
            f2.to_pairs(),
            (
                (('a', 1), (('x', 1), ('y', 30), ('z', 54))),
                (('a', 2), (('x', 2), ('y', 34), ('z', 95))),
                (('b', 1), (('x', 'a'), ('y', 'b'), ('z', 'c'))),
                (('b', 2), (('x', False), ('y', True), ('z', False))),
                (('b', 3), (('x', True), ('y', False), ('z', False))),
            ),
        )

    def test_frame_relabel_flat_b(self) -> None:
        records = (
            (2, 'a', False),
            (34, 'b', True),
        )
        f1 = Frame.from_records(records, columns=('p', 'q', 'r'), index=('w', 'x'))

        with self.assertRaises(RuntimeError):
            _ = f1.relabel_flat()

    # ---------------------------------------------------------------------------

    def test_frame_relabel_add_level_a(self) -> None:
        f1 = sf.FrameGO.from_element(1, index=[1, 2, 3], columns=['a'])
        f2 = f1.relabel_level_add(columns='a')

        self.assertEqual(f2.columns.__class__, IndexHierarchyGO)
        self.assertEqual(f2.columns.values.tolist(), [['a', 'a']])

    def test_frame_relabel_add_level_b(self) -> None:
        f1 = sf.FrameGO.from_element(1, index=[1, 2, 3], columns=['a'])
        f2 = f1.relabel_level_add(
            columns='1542',
            index='2050',
            index_constructor=IndexYear,
            columns_constructor=IndexYear,
        )
        self.assertEqual(f2.index.index_types.values.tolist(), [IndexYear, Index])
        self.assertEqual(f2.columns.index_types.values.tolist(), [IndexYearGO, IndexGO])

    # ---------------------------------------------------------------------------

    def test_frame_rename_a(self) -> None:
        records = (
            (2, 'a', False),
            (34, 'b', True),
        )
        f1 = Frame.from_records(
            records, columns=('p', 'q', 'r'), index=('w', 'x'), name='foo'
        )
        self.assertEqual(f1.name, 'foo')

        f2 = Frame(f1)
        self.assertEqual(f2.name, 'foo')

    def test_frame_rename_b(self) -> None:
        records = (
            (2, 'a', False),
            (34, 'b', True),
        )
        f1 = Frame.from_records(
            records, columns=('p', 'q', 'r'), index=('w', 'x'), name='foo'
        )

        f2 = f1.rename(None, index='a', columns='b')
        self.assertEqual(f2.name, None)
        self.assertEqual(f2.index.name, 'a')
        self.assertEqual(f2.columns.name, 'b')

    def test_frame_rename_c(self) -> None:
        records = (
            (2, 'a', False),
            (34, 'b', True),
        )
        f1 = Frame.from_records(
            records, columns=('p', 'q', 'r'), index=('w', 'x'), name='foo'
        )

        f2 = f1.rename(columns='x')
        self.assertEqual(f2.name, 'foo')
        self.assertEqual(f2.columns.name, 'x')

    # ---------------------------------------------------------------------------

    def test_frame_add_level_a(self) -> None:
        records = (
            (1, 2, 'a', False, True),
            (30, 34, 'b', True, False),
            (54, 95, 'c', False, False),
        )
        f1 = Frame.from_records(
            records, columns=('a', 'b', 'c', 'd', 'e'), index=('x', 'y', 'z')
        )

        f2 = f1.relabel_level_add(index='I', columns='II')

        self.assertEqual(
            f2.to_pairs(),
            (
                (('II', 'a'), ((('I', 'x'), 1), (('I', 'y'), 30), (('I', 'z'), 54))),
                (('II', 'b'), ((('I', 'x'), 2), (('I', 'y'), 34), (('I', 'z'), 95))),
                (
                    ('II', 'c'),
                    ((('I', 'x'), 'a'), (('I', 'y'), 'b'), (('I', 'z'), 'c')),
                ),
                (
                    ('II', 'd'),
                    ((('I', 'x'), False), (('I', 'y'), True), (('I', 'z'), False)),
                ),
                (
                    ('II', 'e'),
                    ((('I', 'x'), True), (('I', 'y'), False), (('I', 'z'), False)),
                ),
            ),
        )

    # ---------------------------------------------------------------------------

    def test_frame_from_from_pandas_a(self) -> None:
        import pandas as pd

        pdf = pd.DataFrame(
            dict(
                a=(False, True, False),
                b=(False, False, False),
                c=(1, 2, 3),
                d=(4, 5, 6),
                e=(None, None, None),
            )
        )

        sff = Frame.from_pandas(pdf)
        self.assertTrue((pdf.dtypes.values == sff.dtypes.values).all())

    def test_frame_from_from_pandas_b(self) -> None:
        import pandas as pd

        f = Frame.from_pandas(pd.DataFrame())
        self.assertEqual(f.shape, (0, 0))

    def test_frame_from_from_pandas_c(self) -> None:
        import pandas as pd

        pdf = pd.DataFrame(
            dict(a=(False, True, False), b=(1, 2, 3)),
        )
        f1 = Frame.from_pandas(
            pdf,
            index=(10, 20, 30),
            columns=('x', 'y'),
        )
        self.assertEqual(f1.index.values.tolist(), [10, 20, 30])
        self.assertEqual(f1.columns.values.tolist(), ['x', 'y'])

        f2 = Frame.from_pandas(
            pdf,
            index=IndexAutoFactory,
            columns=IndexAutoFactory,
        )
        self.assertEqual(f2.index._map, None)  # type: ignore
        self.assertEqual(f2.columns._map, None)  # type: ignore

    # ---------------------------------------------------------------------------

    def test_frame_to_frame_go_a(self) -> None:
        records = (
            (1, 2, 'a', False, True),
            (30, 34, 'b', True, False),
            (54, 95, 'c', False, False),
        )
        f1 = Frame.from_records(
            records, columns=('a', 'b', 'c', 'd', 'e'), index=('x', 'y', 'z')
        )

        f2 = f1.to_frame_go()
        f2['f'] = None
        self.assertEqual(f1.columns.values.tolist(), ['a', 'b', 'c', 'd', 'e'])
        self.assertEqual(f2.columns.values.tolist(), ['a', 'b', 'c', 'd', 'e', 'f'])

        # underlying map objects must be different
        self.assertTrue(id(f1.columns._map) != id(f2.columns._map))  # type: ignore

    def test_frame_to_frame_go_b(self) -> None:
        records = (
            (1, 2, 'a', False, True),
            (54, 95, 'c', False, False),
        )
        f1 = Frame.from_records(
            records, columns=('a', 'b', 'c', 'd', 'e'), index=('x', 'y'), name='foo'
        )

        f2 = f1.to_frame_go()
        f3 = f2.to_frame(name='bar')

        self.assertTrue(f1.name, 'foo')
        self.assertTrue(f2.name, 'foo')
        self.assertTrue(f3.name, 'bar')

    def test_frame_to_frame_go_c(self) -> None:
        records = (
            (1, 'a', False, True),
            (1, 'b', False, False),
            (2, 'a', False, True),
            (2, 'b', False, False),
        )
        f1 = Frame.from_records(
            records,
            columns=IndexHierarchy.from_product((1, 2), ('a', 'b')),
            index=('w', 'x', 'y', 'z'),
            name='foo',
        )

        f2 = f1.to_frame_go()

        self.assertTrue(isinstance(f2.columns, IndexHierarchyGO))

        f2[(3, 'a')] = 10

        self.assertEqual(
            f2.to_pairs(),
            (
                ((1, 'a'), (('w', 1), ('x', 1), ('y', 2), ('z', 2))),
                ((1, 'b'), (('w', 'a'), ('x', 'b'), ('y', 'a'), ('z', 'b'))),
                ((2, 'a'), (('w', False), ('x', False), ('y', False), ('z', False))),
                ((2, 'b'), (('w', True), ('x', False), ('y', True), ('z', False))),
                ((3, 'a'), (('w', 10), ('x', 10), ('y', 10), ('z', 10))),
            ),
        )

    def test_frame_to_frame_go_d(self) -> None:
        records = (
            (2, 'a', False),
            (34, 'b', True),
        )
        f1 = FrameGO.from_records(records, columns=('p', 'q', 'r'), index=('w', 'x'))

        f2 = f1.to_frame_go()

        f1['x'] = None
        f2['a'] = -1

        self.assertEqual(
            f1.to_pairs(),
            (
                ('p', (('w', 2), ('x', 34))),
                ('q', (('w', 'a'), ('x', 'b'))),
                ('r', (('w', False), ('x', True))),
                ('x', (('w', None), ('x', None))),
            ),
        )
        self.assertEqual(
            f2.to_pairs(),
            (
                ('p', (('w', 2), ('x', 34))),
                ('q', (('w', 'a'), ('x', 'b'))),
                ('r', (('w', False), ('x', True))),
                ('a', (('w', -1), ('x', -1))),
            ),
        )

    def test_frame_to_frame_go_e(self) -> None:
        records = (
            (2, 'a', False),
            (34, 'b', True),
        )
        f1 = Frame.from_records(records, columns=('p', 'q', 'r'), index=('w', 'x'))

        f2 = f1.to_frame()
        f3 = f1.to_frame_go()

        self.assertTrue(id(f1) == id(f2))
        self.assertTrue(id(f1) != id(f3))

        f3['x'] = None

        self.assertEqual(
            f3.to_pairs(),
            (
                ('p', (('w', 2), ('x', 34))),
                ('q', (('w', 'a'), ('x', 'b'))),
                ('r', (('w', False), ('x', True))),
                ('x', (('w', None), ('x', None))),
            ),
        )

    # ---------------------------------------------------------------------------

    def test_frame_to_frame_a(self) -> None:
        f1 = Frame(np.array(()), name='foo')
        f2 = f1.to_frame(name='bar')
        self.assertEqual(f2.name, 'bar')

    # ---------------------------------------------------------------------------

    def test_frame_to_frame_he_a(self) -> None:
        records = (
            (2, 'a', False),
            (34, 'b', True),
        )
        f1 = Frame.from_records(records, columns=('p', 'q', 'r'), index=('w', 'x'))
        f2 = f1.to_frame_he()
        post = {f2, f2}
        self.assertEqual(len(post), 1)

        post.add(f2.to_frame_he())
        self.assertEqual(len(post), 1)

        f3 = f2.to_frame()
        self.assertIs(f3.__class__, Frame)

        f4 = f2.to_frame_go()
        f4['s'] = None

        self.assertEqual(
            f4.to_pairs(),
            (
                ('p', (('w', 2), ('x', 34))),
                ('q', (('w', 'a'), ('x', 'b'))),
                ('r', (('w', False), ('x', True))),
                ('s', (('w', None), ('x', None))),
            ),
        )

    def test_frame_to_frame_he_b(self) -> None:
        records = (
            (2, 'a', False),
            (34, 'b', True),
        )
        f1 = FrameGO.from_records(records, columns=('p', 'q', 'r'), index=('w', 'x'))
        f2 = f1.to_frame_he()
        self.assertIs(f2.__class__, FrameHE)
        self.assertEqual(
            f2.to_pairs(),
            (
                ('p', (('w', 2), ('x', 34))),
                ('q', (('w', 'a'), ('x', 'b'))),
                ('r', (('w', False), ('x', True))),
            ),
        )

    def test_frame_to_frame_he_c(self) -> None:
        fhe1 = ff.parse('f(Fg)|v(int,bool,str)|i((I,I),(str,str))|s(5,4)').to_frame_he()
        fhe2 = ff.parse(
            'f(Fg)|v(int,bool,str)|c((Isg,Isg),(dts, dts))|s(5,4)'
        ).to_frame_he()
        post = {}
        post[fhe1] = 0
        post[fhe2] = 1
        self.assertTrue(fhe1 in post)
        self.assertTrue(fhe2 in post)

    # ---------------------------------------------------------------------------

    def test_frame_to_npz_a(self) -> None:
        f1 = ff.parse('s(10_000,2)|v(int,str)|i((I, ID),(str,dtD))|c(ID,dtD)').rename(
            'foo'
        )

        with temp_file('.npz') as fp:
            f1.to_npz(fp)
            f2 = Frame.from_npz(fp)
            self.assertTrue(
                f1.equals(f2, compare_dtype=True, compare_class=True, compare_name=True)
            )

    def test_frame_to_npz_b(self) -> None:
        f1 = ff.parse('s(10_000,2)|v(int,str)|c((I, ID),(str,dtD))|i(ID,dtD)').rename(
            'foo'
        )

        with temp_file('.npz') as fp:
            f1.to_npz(fp)
            f2 = Frame.from_npz(fp)
            self.assertTrue(
                f1.equals(f2, compare_dtype=True, compare_class=True, compare_name=True)
            )

    def test_frame_to_npz_c(self) -> None:
        f1 = ff.parse('s(20,100)|v(int,str,bool)').rename('foo')

        with temp_file('.npz') as fp:
            f1.to_npz(fp)
            f2 = Frame.from_npz(fp)
            self.assertTrue(
                f1.equals(f2, compare_dtype=True, compare_class=True, compare_name=True)
            )

    @pytest.mark.skip('failing due to index name mismatch')
    def test_frame_to_npz_d(self) -> None:
        f1 = ff.parse('s(10,100)|v(int,str,bool,bool,float,float)').rename(
            'foo', index='bar', columns='baz'
        )

        with temp_file('.npz') as fp:
            f1.to_npz(fp)
            f2 = Frame.from_npz(fp)
            self.assertTrue(
                f1.equals(f2, compare_dtype=True, compare_class=True, compare_name=True)
            )

    def test_frame_to_npz_e(self) -> None:
        f1 = ff.parse('s(10,100)|v(bool,bool,float,float)|i(I,str)|c(I,int)').rename(
            'foo', index='bar', columns='baz'
        )

        with temp_file('.npz') as fp:
            f1.to_npz(fp)
            f2 = Frame.from_npz(fp)
            self.assertTrue(
                f1.equals(f2, compare_dtype=True, compare_class=True, compare_name=True)
            )

    def test_frame_to_npz_f(self) -> None:
        f1 = ff.parse(
            's(10,100)|v(bool,bool,float,float)|i((ID,IY),(dtD,dtY))|c((IY,I),(dtY,str))'
        ).rename('foo', index=('a', 'b'), columns=('x', 'y'))

        with temp_file('.npz') as fp:
            f1.to_npz(fp)
            f2 = Frame.from_npz(fp)
            self.assertTrue(
                f1.equals(f2, compare_dtype=True, compare_class=True, compare_name=True)
            )

    def test_frame_to_npz_g(self) -> None:
        f1 = ff.parse('s(20,100)|v(int,str,bool)').rename('foo')

        with temp_file('.npz') as fp:
            f1.to_npz(fp)
            f2 = Frame.from_npz(fp)
            self.assertTrue(
                f1.equals(f2, compare_dtype=True, compare_class=True, compare_name=True)
            )

    def test_frame_to_npz_h(self) -> None:
        f1 = ff.parse('s(20,100)|v(int,str,bool)').rename(((1, 2), (3, 4)))

        with temp_file('.npz') as fp:
            f1.to_npz(fp)
            f2 = Frame.from_npz(fp)
            self.assertTrue(
                f1.equals(f2, compare_dtype=True, compare_class=True, compare_name=True)
            )

    def test_frame_to_npz_i(self) -> None:
        f1 = ff.parse('s(20,100)|v(int,str)').rename(((1, 2), (3, 4)))
        f2 = f1[f1.dtypes == int]  # force maximally partitioned

        with temp_file('.npz') as fp:
            f2.to_npz(fp, consolidate_blocks=True)
            f3 = Frame.from_npz(fp)
            self.assertTrue(
                f2.equals(f3, compare_dtype=True, compare_class=True, compare_name=True)
            )
            self.assertEqual(f3._blocks.shapes.tolist(), [(20, 50)])

    def test_frame_to_npz_j(self) -> None:
        f1 = ff.parse('s(100,2)|v(int,object)')

        with temp_file('.npz') as fp:
            try:
                f1.to_npz(fp)
            except ErrorNPYEncode:
                pass

    def test_frame_to_npz_k(self) -> None:
        f1 = ff.parse('s(100,2)|v(int)')

        with temp_file('.npz') as fp:
            f1.to_npz(fp)
            f2 = FrameGO.from_npz(fp)
            self.assertTrue(f1.equals(f2))
            self.assertIs(f2.__class__, FrameGO)

    def test_frame_to_npz_l1(self) -> None:
        f1 = ff.parse('s(10,6)|v(str)').rename(np.datetime64('2022-02-22'))

        with temp_file('.npz') as fp:
            f1.to_npz(fp)
            f2 = Frame.from_npz(fp)
            self.assertEqual(f1.name, f2.name)
            self.assertTrue(f1.equals(f2))

    def test_frame_to_npz_l2(self) -> None:
        f1 = ff.parse('s(10,6)|v(str)').rename(datetime.date(2022, 2, 22))

        with temp_file('.npz') as fp:
            f1.to_npz(fp)
            f2 = Frame.from_npz(fp)
            self.assertEqual(f1.name, f2.name)
            self.assertTrue(f1.equals(f2))

    def test_frame_to_npz_empty(self) -> None:
        f1 = Frame()

        with temp_file('.npz') as fp:
            f1.to_npz(fp)
            f2 = Frame.from_npz(fp)
            self.assertTrue(f1.equals(f2))

    def test_frame_to_npz_bytesio(self) -> None:
        f1: Frame = ff.parse('s(10,6)|v(str)').rename('foo')

        with TemporaryFile() as tf:
            f1.to_npz(tf)
            f2 = sf.Frame.from_npz(tf)
            self.assertEqual(f1.name, f2.name)
            self.assertTrue(f1.equals(f2))

    def test_frame_to_npz_failure_a(self) -> None:
        from datetime import date

        with temp_file('.npz') as fp:
            with self.assertRaises(ErrorNPYEncode):
                # fails to being an object array
                sf.Series([1, 2, 3], name=date(2022, 1, 1)).to_frame().to_npz(fp)
            self.assertFalse(os.path.exists(fp))

    # ---------------------------------------------------------------------------

    def test_frame_from_npz_a(self) -> None:
        f1 = ff.parse('f(Fg)|v(int,bool,str)|i((IY,ID),(dtY,dtD))|c(Isg,dts)|s(6,2)')
        dt64 = np.datetime64
        with temp_file('.npz') as fp:
            f1.to_npz(fp)
            f2 = sf.Frame.from_npz(fp)
            self.assertIs(f2.columns.__class__, IndexSecond)
            self.assertEqual(
                f2.to_pairs(),
                (
                    (
                        dt64('1970-01-01T09:38:35'),
                        (
                            ((dt64('36685'), dt64('2258-03-21')), -88017),
                            ((dt64('36685'), dt64('2298-04-20')), 92867),
                            ((dt64('5618'), dt64('2501-10-08')), 84967),
                            ((dt64('5618'), dt64('2441-04-14')), 13448),
                            ((dt64('93271'), dt64('2234-04-07')), 175579),
                            ((dt64('93271'), dt64('2210-12-26')), 58768),
                        ),
                    ),
                    (
                        dt64('1970-01-01T01:00:48'),
                        (
                            ((dt64('36685'), dt64('2258-03-21')), False),
                            ((dt64('36685'), dt64('2298-04-20')), False),
                            ((dt64('5618'), dt64('2501-10-08')), False),
                            ((dt64('5618'), dt64('2441-04-14')), False),
                            ((dt64('93271'), dt64('2234-04-07')), False),
                            ((dt64('93271'), dt64('2210-12-26')), False),
                        ),
                    ),
                ),
            )

    # ---------------------------------------------------------------------------

    def test_frame_to_npy_a(self) -> None:
        f1 = ff.parse('s(10_000,2)|v(int,str)|i((I, ID),(str,dtD))|c(ID,dtD)').rename(
            'foo'
        )
        with TemporaryDirectory() as fp:
            os.rmdir(fp)  # let it be re-created
            f1.to_npy(fp)
            f2 = Frame.from_npy(fp)
            f1.equals(f2, compare_dtype=True, compare_class=True, compare_name=True)

    def test_frame_to_npy_b(self) -> None:
        f1 = ff.parse('s(10_000,2)|v(int,str)|c((I, ID),(str,dtD))|i(ID,dtD)').rename(
            'foo'
        )

        with TemporaryDirectory() as fp:
            os.rmdir(fp)  # let it be re-created
            f1.to_npy(fp)
            f2 = Frame.from_npy(fp)
            f1.equals(f2, compare_dtype=True, compare_class=True, compare_name=True)

    def test_frame_to_npy_c(self) -> None:
        f1 = ff.parse('s(20,100)|v(int,str,bool)').rename('foo')

        with TemporaryDirectory() as fp:
            os.rmdir(fp)  # let it be re-created
            f1.to_npy(fp)
            f2 = Frame.from_npy(fp)
            f1.equals(f2, compare_dtype=True, compare_class=True, compare_name=True)

    def test_frame_to_npy_d(self) -> None:
        f1 = ff.parse('s(10,100)|v(int,str,bool,bool,float,float)').rename(
            'foo', index='bar', columns='baz'
        )

        with TemporaryDirectory() as fp:
            os.rmdir(fp)  # let it be re-created
            f1.to_npy(fp)
            f2 = Frame.from_npy(fp)
            f1.equals(f2, compare_dtype=True, compare_class=True, compare_name=True)

    def test_frame_to_npy_e(self) -> None:
        f1 = ff.parse('s(10,100)|v(bool,bool,float,float)|i(I,str)|c(I,int)').rename(
            'foo', index='bar', columns='baz'
        )

        with TemporaryDirectory() as fp:
            os.rmdir(fp)  # let it be re-created
            f1.to_npy(fp)
            f2 = Frame.from_npy(fp)
            f1.equals(f2, compare_dtype=True, compare_class=True, compare_name=True)

    def test_frame_to_npy_f(self) -> None:
        f1 = ff.parse(
            's(10,100)|v(bool,bool,float,float)|i((ID,IY),(dtD,dtY))|c((IY,I),(dtY,str))'
        ).rename('foo', index=('a', 'b'), columns=('x', 'y'))

        with TemporaryDirectory() as fp:
            os.rmdir(fp)  # let it be re-created
            f1.to_npy(fp)
            f2 = Frame.from_npy(fp)
            f1.equals(f2, compare_dtype=True, compare_class=True, compare_name=True)

    def test_frame_to_npy_g(self) -> None:
        f1 = ff.parse('s(20,100)|v(int,str,bool)').rename('foo')

        with TemporaryDirectory() as fp:
            os.rmdir(fp)  # let it be re-created
            f1.to_npy(fp)
            f2 = Frame.from_npy(fp)
            f1.equals(f2, compare_dtype=True, compare_class=True, compare_name=True)

    def test_frame_to_npy_h(self) -> None:
        f1 = ff.parse('s(20,100)|v(int,str,bool)').rename(((1, 2), (3, 4)))

        with TemporaryDirectory() as fp:
            os.rmdir(fp)  # let it be re-created
            f1.to_npy(fp)
            f2 = Frame.from_npy(fp)
            f1.equals(f2, compare_dtype=True, compare_class=True, compare_name=True)

    def test_frame_to_npy_i(self) -> None:
        f1 = ff.parse('s(20,100)|v(int,str)').rename(((1, 2), (3, 4)))
        f2 = f1[f1.dtypes == int]  # force maximally partitioned

        with TemporaryDirectory() as fp:
            os.rmdir(fp)  # let it be re-created
            f2.to_npy(fp, consolidate_blocks=True)
            f3 = Frame.from_npy(fp)
            f2.equals(f3, compare_dtype=True, compare_class=True, compare_name=True)
            self.assertEqual(f3._blocks.shapes.tolist(), [(20, 50)])

    def test_frame_to_npy_failure_a(self) -> None:
        from datetime import date

        with TemporaryDirectory() as fp:
            os.rmdir(fp)  # let it be re-created
            with self.assertRaises(ErrorNPYEncode):
                sf.Series([1, 2, 3], name=date(2022, 1, 1)).to_frame().to_npy(fp)
            self.assertFalse(os.path.exists(fp))
            # restore so test does not complain
            os.mkdir(fp)

    # ---------------------------------------------------------------------------

    def test_frame_from_npy_memory_map_a(self) -> None:
        f1 = ff.parse('s(10_000,2)|v(int,str)|i((I, ID),(str,dtD))|c(ID,dtD)').rename(
            'foo'
        )
        with TemporaryDirectory() as fp:
            os.rmdir(fp)  # let it be re-created
            f1.to_npy(fp)
            f2, finalizer = Frame.from_npy_mmap(fp)
            f1.equals(f2, compare_dtype=True, compare_class=True, compare_name=True)
            finalizer()

    def test_frame_from_npy_memory_map_b(self) -> None:
        f1 = ff.parse('s(10_000,2)|v(int,str)|i((I, ID),(str,dtD))|c(ID,dtD)').rename(
            'foo'
        )
        with TemporaryDirectory() as fp:
            os.rmdir(fp)  # let it be re-created
            f1.to_npy(fp)
            f2, finalizer = Frame.from_npy_mmap(fp)
            f3 = f2.to_frame()
            finalizer()

    def test_frame_from_npy_memory_map_c(self) -> None:
        f1 = ff.parse('s(3,3)')
        with TemporaryDirectory() as fp:
            os.rmdir(fp)  # let it be re-created
            f1.to_npy(fp)
            f1, finalizer = sf.Frame.from_npy_mmap(fp)
            f1 = f1.set_index(0)
            post = str(f1)
            finalizer()

    def test_frame_from_npy_memory_map_d(self) -> None:
        with TemporaryDirectory() as fp:
            os.rmdir(fp)  # let it be re-created

            class C:
                def __init__(self) -> None:
                    ff.parse('s(3,3)').to_npy(fp)
                    self.finalizer: tp.Optional[tp.Callable[[], None]] = None
                    self.frame: tp.Optional[sf.Frame] = None

                def __del__(self) -> None:
                    self.finalizer()

                @property
                def mmap_frame(self) -> sf.Frame:
                    if not self.finalizer:
                        self.frame, self.finalizer = Frame.from_npy_mmap(fp)
                    return self.frame  # type: ignore

            c = C()
            f1 = c.mmap_frame
            self.assertEqual(f1.shape, (3, 3))

            s1 = c.mmap_frame.iloc[0]
            self.assertEqual(
                round(s1, 1).to_pairs(), ((0, 1930.4), (1, -610.8), (2, 694.3))
            )
            c.__del__()

    def test_frame_from_npy_memory_map_e(self) -> None:
        f1 = ff.parse('s(3,3)|v(str,bool)').astype[2](object)
        with TemporaryDirectory() as fp:
            os.rmdir(fp)  # let it be re-created
            with self.assertRaises(ErrorNPYEncode):
                f1.to_npy(fp)
            self.assertFalse(os.path.exists(fp))
            os.mkdir(fp)

    # ---------------------------------------------------------------------------

    def test_frame_astype_a(self) -> None:
        records = (
            (1, 2, 'a', False, True),
            (30, 34, 'b', True, False),
            (54, 95, 'c', False, False),
        )
        f1 = Frame.from_records(
            records, columns=('a', 'b', 'c', 'd', 'e'), index=('x', 'y', 'z')
        )

        f2 = f1.astype['d':](int)  # type: ignore  # https://github.com/python/typeshed/pull/3024
        self.assertEqual(
            f2.to_pairs(),
            (
                ('a', (('x', 1), ('y', 30), ('z', 54))),
                ('b', (('x', 2), ('y', 34), ('z', 95))),
                ('c', (('x', 'a'), ('y', 'b'), ('z', 'c'))),
                ('d', (('x', 0), ('y', 1), ('z', 0))),
                ('e', (('x', 1), ('y', 0), ('z', 0))),
            ),
        )

        f3 = f1.astype[['a', 'b']](bool)
        self.assertEqual(
            f3.to_pairs(),
            (
                ('a', (('x', True), ('y', True), ('z', True))),
                ('b', (('x', True), ('y', True), ('z', True))),
                ('c', (('x', 'a'), ('y', 'b'), ('z', 'c'))),
                ('d', (('x', False), ('y', True), ('z', False))),
                ('e', (('x', True), ('y', False), ('z', False))),
            ),
        )

    def test_frame_astype_b(self) -> None:
        records = (
            (1, 2, 'a', False, True),
            (30, 34, 'b', True, False),
            (54, 95, 'c', False, False),
        )
        f1 = Frame.from_records(
            records, columns=('a', 'b', 'c', 'd', 'e'), index=('x', 'y', 'z')
        )

        f2 = f1.astype({'b': str, 'e': str})

        self.assertEqual([dt.kind for dt in f2.dtypes.values], ['i', 'U', 'U', 'b', 'U'])

        f3 = f1.astype[:]({'b': str, 'e': str})
        self.assertEqual([dt.kind for dt in f3.dtypes.values], ['i', 'U', 'U', 'b', 'U'])

        with self.assertRaises(RuntimeError):
            _ = f1.astype['c':]({'b': str, 'e': str})  # type: ignore

    def test_frame_astype_c(self) -> None:
        records = (
            (1, 2, 'a', False, True),
            (30, 34, 'b', True, False),
            (54, 95, 'c', False, False),
        )
        f1 = Frame.from_records(
            records, columns=('a', 'b', 'c', 'd', 'e'), index=('x', 'y', 'z')
        )

        f2 = f1.astype((float, float, None, int, int))
        self.assertEqual([dt.kind for dt in f2.dtypes.values], ['f', 'f', 'U', 'i', 'i'])

        f3 = f1.astype(str)
        self.assertEqual([dt.kind for dt in f3.dtypes.values], ['U', 'U', 'U', 'U', 'U'])

    def test_frame_astype_d(self) -> None:
        f1 = ff.parse('s(3,5)')
        f2 = f1.astype[[1, 2, 3]](int)
        f3 = f2.astype[f2.dtypes == int](float)
        self.assertEqual([dt.kind for dt in f3.dtypes.values], ['f', 'f', 'f', 'f', 'f'])

        f4 = f3.astype[2](int)
        self.assertEqual([dt.kind for dt in f4.dtypes.values], ['f', 'f', 'i', 'f', 'f'])

    def test_frame_astype_e(self) -> None:
        f1 = ff.parse('s(3,5)')
        f2 = f1.astype[[1, 2, 3]](int)
        f3 = f2.astype[(f2.dtypes == int).values](float)
        self.assertEqual([dt.kind for dt in f3.dtypes.values], ['f', 'f', 'f', 'f', 'f'])

    def test_frame_astype_f(self) -> None:
        f1 = Frame(np.arange(25).reshape(5, 5), columns=[0, 1, 2, 3, 4])
        f2 = f1.astype({'a': bool, 0: bool})
        self.assertEqual(
            f2.to_pairs(),
            (
                (0, ((0, False), (1, True), (2, True), (3, True), (4, True))),
                (1, ((0, 1), (1, 6), (2, 11), (3, 16), (4, 21))),
                (2, ((0, 2), (1, 7), (2, 12), (3, 17), (4, 22))),
                (3, ((0, 3), (1, 8), (2, 13), (3, 18), (4, 23))),
                (4, ((0, 4), (1, 9), (2, 14), (3, 19), (4, 24))),
            ),
        )

    def test_frame_astype_g(self) -> None:
        f1 = ff.parse('s(3,5)|v(int,str,bool,str,int)')
        f2 = f1.astype[f1.dtypes.values == int](str)
        self.assertEqual([d.kind for d in f2.dtypes.values], ['U', 'U', 'b', 'U', 'U'])

    def test_frame_astype_h(self) -> None:
        f1 = Frame().astype(str)
        self.assertEqual(f1.shape, (0, 0))

        f2 = Frame().astype[:](str)
        self.assertEqual(f2.shape, (0, 0))

    def test_frame_astype_i(self) -> None:
        f1 = Frame.from_fields(
            ((10, 20), (30, 40), (50, 60), (70, 80)), columns=('a', 'b', 'c', 'd')
        )
        self.assertEqual(len(f1._blocks._blocks), 4)

        f2 = f1.astype[['c', 'd']](float, consolidate_blocks=True)
        self.assertEqual(f2._blocks.shapes.tolist(), [(2, 2), (2, 2)])

    def test_frame_astype_j(self) -> None:
        f1 = Frame.from_fields(
            ((10, 20), (30, 40), (50, 60), (70, 80)), columns=('a', 'b', 'c', 'd')
        )
        self.assertEqual(len(f1._blocks._blocks), 4)

        f2 = f1.astype[['c', 'd']](float)
        self.assertEqual(f2._blocks.shapes.tolist(), [(2,), (2,), (2,), (2,)])

    def test_frame_astype_k(self) -> None:
        f1 = Frame(np.arange(12).reshape(3, 4))
        with self.assertRaises(KeyError):
            _ = f1.astype[20](float)

    def test_frame_astype_l1(self) -> None:
        records = (
            (1, '2024-01', False),
            (30, '1764-04', True),
            (54, '3030-01', False),
        )
        f1 = Frame.from_records(
            records,
            columns=('a', 'b', 'c'),
            index=('x', 'y', 'z'),
            dtypes=(np.int8, 'datetime64[ns]', np.bool_),
        )
        f2 = f1.astype['b'](object)
        self.assertEqual(
            f2['b'].to_pairs(),
            (
                ('x', np.datetime64('2024-01-01T00:00:00.000000000')),
                ('y', np.datetime64('1764-04-01T00:00:00.000000000')),
                ('z', np.datetime64('1860-11-22T00:50:52.580896768')),
            ),
        )

    def test_frame_astype_l2(self) -> None:
        records = (
            ('2024-01', '1954-01'),
            ('1764-04', '1954-01'),
            ('3030-01', '1954-01'),
        )
        f1 = Frame.from_records(
            records, columns=('a', 'b'), index=('x', 'y', 'z'), dtypes='datetime64[ns]'
        )
        f2 = f1.astype(object)
        self.assertEqual(
            f2.to_pairs(),
            (
                (
                    'a',
                    (
                        ('x', np.datetime64('2024-01-01T00:00:00.000000000')),
                        ('y', np.datetime64('1764-04-01T00:00:00.000000000')),
                        ('z', np.datetime64('1860-11-22T00:50:52.580896768')),
                    ),
                ),
                (
                    'b',
                    (
                        ('x', np.datetime64('1954-01-01T00:00:00.000000000')),
                        ('y', np.datetime64('1954-01-01T00:00:00.000000000')),
                        ('z', np.datetime64('1954-01-01T00:00:00.000000000')),
                    ),
                ),
            ),
        )

    # ---------------------------------------------------------------------------

    def test_frame_pickle_a(self) -> None:
        records = ((1, 2, 'a', False, True), (30, 50, 'b', True, False))

        f1 = FrameGO.from_records(
            records, columns=('p', 'q', 'r', 's', 't'), index=('x', 'y')
        )

        pbytes = pickle.dumps(f1)
        f2 = pickle.loads(pbytes)

        self.assertEqual(
            [b.flags.writeable for b in f2._blocks._blocks],
            [False, False, False, False, False],
        )

    # ---------------------------------------------------------------------------

    def test_frame_to_pickle_a(self) -> None:
        f1 = ff.parse('s(10_000,2)|v(int,str)|i((I, ID),(str,dtD))|c(ID,dtD)').rename(
            'foo'
        )

        with temp_file('.pickle') as fp:
            f1.to_pickle(fp)
            f2 = Frame.from_pickle(fp)
            f1.equals(f2, compare_dtype=True, compare_class=True, compare_name=True)

    def test_frame_to_pickle_b(self) -> None:
        f1 = ff.parse('s(10,2_000)|v(int,str,bool)|i((I, ID),(str,dtD))|c(ID,dtD)')

        with temp_file('.pickle') as fp:
            f1.to_pickle(fp)
            f2 = Frame.from_pickle(fp)
            f1.equals(f2, compare_dtype=True, compare_class=True, compare_name=True)

            f3 = FrameGO.from_pickle(fp)
            f1.equals(f3, compare_dtype=True, compare_class=False, compare_name=True)
            self.assertIs(f3.__class__, FrameGO)

            f4 = FrameHE.from_pickle(fp)
            f1.equals(f4, compare_dtype=True, compare_class=False, compare_name=True)
            self.assertIs(f4.__class__, FrameHE)

    # ---------------------------------------------------------------------------

    def test_frame_set_index_hierarchy_a(self) -> None:
        records = (
            (1, 2, 'a', False, True),
            (30, 2, 'b', True, False),
            (30, 50, 'a', True, False),
            (30, 50, 'b', True, False),
        )

        f1 = FrameGO.from_records(
            records, columns=('p', 'q', 'r', 's', 't'), index=('w', 'x', 'y', 'z')
        )

        f2 = f1.set_index_hierarchy(['q', 'r'])
        self.assertEqual(f2.index.name, ('q', 'r'))

        # we reuse the same block data
        self.assertTrue((f2.index._blocks.mloc == f1._blocks[[1, 2]].mloc).all())  # type: ignore

        self.assertEqual(
            f2.to_pairs(),
            (
                (
                    'p',
                    (((2, 'a'), 1), ((2, 'b'), 30), ((50, 'a'), 30), ((50, 'b'), 30)),
                ),
                ('q', (((2, 'a'), 2), ((2, 'b'), 2), ((50, 'a'), 50), ((50, 'b'), 50))),
                (
                    'r',
                    (
                        ((2, 'a'), 'a'),
                        ((2, 'b'), 'b'),
                        ((50, 'a'), 'a'),
                        ((50, 'b'), 'b'),
                    ),
                ),
                (
                    's',
                    (
                        ((2, 'a'), False),
                        ((2, 'b'), True),
                        ((50, 'a'), True),
                        ((50, 'b'), True),
                    ),
                ),
                (
                    't',
                    (
                        ((2, 'a'), True),
                        ((2, 'b'), False),
                        ((50, 'a'), False),
                        ((50, 'b'), False),
                    ),
                ),
            ),
        )

        f3 = f1.set_index_hierarchy(['q', 'r'], drop=True)
        self.assertEqual(f3.index.name, ('q', 'r'))

        self.assertEqual(
            f3.to_pairs(),
            (
                (
                    'p',
                    (((2, 'a'), 1), ((2, 'b'), 30), ((50, 'a'), 30), ((50, 'b'), 30)),
                ),
                (
                    's',
                    (
                        ((2, 'a'), False),
                        ((2, 'b'), True),
                        ((50, 'a'), True),
                        ((50, 'b'), True),
                    ),
                ),
                (
                    't',
                    (
                        ((2, 'a'), True),
                        ((2, 'b'), False),
                        ((50, 'a'), False),
                        ((50, 'b'), False),
                    ),
                ),
            ),
        )

        f4 = f1.set_index_hierarchy(slice('q', 'r'), drop=True)
        self.assertEqual(f4.index.name, ('q', 'r'))

        self.assertEqual(
            f4.to_pairs(),
            (
                (
                    'p',
                    (((2, 'a'), 1), ((2, 'b'), 30), ((50, 'a'), 30), ((50, 'b'), 30)),
                ),
                (
                    's',
                    (
                        ((2, 'a'), False),
                        ((2, 'b'), True),
                        ((50, 'a'), True),
                        ((50, 'b'), True),
                    ),
                ),
                (
                    't',
                    (
                        ((2, 'a'), True),
                        ((2, 'b'), False),
                        ((50, 'a'), False),
                        ((50, 'b'), False),
                    ),
                ),
            ),
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

        f = Frame.from_records(labels)
        f = f.astype[[0, 1]](int)

        fh = f.set_index_hierarchy([0, 1], drop=True)

        self.assertEqual(fh.columns.values.tolist(), [2])

        # we reuse the block arrays in the Index
        self.assertTrue((fh.index._blocks.mloc == f._blocks[:2].mloc).all())  # type: ignore

        self.assertEqual(
            fh.to_pairs(),
            (
                (
                    2,
                    (
                        ((1, 1), 'a'),
                        ((1, 2), 'b'),
                        ((1, 3), 'c'),
                        ((2, 1), 'd'),
                        ((2, 2), 'e'),
                        ((2, 3), 'f'),
                        ((3, 1), 'g'),
                        ((3, 2), 'h'),
                        ((3, 3), 'i'),
                    ),
                ),
            ),
        )

        fh = f.set_index_hierarchy([0, 1], drop=False)
        self.assertEqual(
            fh.loc[HLoc[:, 3]].to_pairs(),
            (
                (0, (((1, 3), 1), ((2, 3), 2), ((3, 3), 3))),
                (1, (((1, 3), 3), ((2, 3), 3), ((3, 3), 3))),
                (2, (((1, 3), 'c'), ((2, 3), 'f'), ((3, 3), 'i'))),
            ),
        )

    def test_frame_set_index_hierarchy_d(self) -> None:
        f1 = sf.Frame.from_records([('one', 1, 'hello')], columns=['name', 'val', 'msg'])

        f2 = f1.set_index_hierarchy(['name', 'val'], drop=True)

        self.assertEqual(f2.to_pairs(), (('msg', ((('one', 1), 'hello'),)),))

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
        fh = f.set_index_hierarchy(
            [0, 1], drop=True, index_constructors=(Index, IndexYearMonth)
        )

        self.assertEqual(
            fh.loc[HLoc[:, '2018']].to_pairs(),
            (
                (
                    2,
                    (
                        ((1, datetime.date(2018, 12, 1)), 10),
                        ((2, datetime.date(2018, 12, 1)), 40),
                    ),
                ),
            ),
        )

        # because we passed index_constructors, we may not be able to reuse blocks
        self.assertTrue((fh.index._blocks.mloc != f._blocks[:2].mloc).all())  # type: ignore

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
        fh = f.set_index_hierarchy(
            [0, 1],
            drop=True,
            reorder_for_hierarchy=True,
        )
        self.assertEqual(
            fh.to_pairs(),
            (
                (
                    2,
                    (
                        ((1, 'a'), 10),
                        ((1, 'c'), 30),
                        ((1, 'b'), 20),
                        ((2, 'a'), 40),
                        ((2, 'c'), 60),
                        ((2, 'b'), 50),
                    ),
                ),
            ),
        )

    def test_frame_set_index_hierarchy_g(self) -> None:
        f1 = ff.parse('s(3,4)|v(dtD, dtY, str, str)')

        f2 = f1.set_index_hierarchy(
            (0, 1),
            index_constructors=IndexAutoConstructorFactory,
            drop=True,
        )
        dt64 = np.datetime64
        self.assertEqual(
            f2.to_pairs(),
            (
                (
                    2,
                    (
                        ((dt64('2210-12-26'), dt64('164167')), 'ztsv'),
                        ((dt64('2224-04-06'), dt64('43127')), 'zUvW'),
                        ((dt64('2202-08-20'), dt64('7699')), 'zkuW'),
                    ),
                ),
                (
                    3,
                    (
                        ((dt64('2210-12-26'), dt64('164167')), 'z2Oo'),
                        ((dt64('2224-04-06'), dt64('43127')), 'z5l6'),
                        ((dt64('2202-08-20'), dt64('7699')), 'zCE3'),
                    ),
                ),
            ),
        )
        self.assertEqual(f2.index.name, (0, 1))

    def test_frame_set_index_hierarchy_h(self) -> None:
        IACF = IndexAutoConstructorFactory
        f1 = ff.parse('s(3,4)|v(dtD, dtY, str, str)')
        f2 = f1.set_index_hierarchy(
            (0, 1),
            index_constructors=(IACF('a'), IACF('b')),
            drop=True,
        )
        self.assertEqual(f2.index.name, ('a', 'b'))
        dt64 = np.datetime64
        self.assertEqual(
            f2.to_pairs(),
            (
                (
                    2,
                    (
                        ((dt64('2210-12-26'), dt64('164167')), 'ztsv'),
                        ((dt64('2224-04-06'), dt64('43127')), 'zUvW'),
                        ((dt64('2202-08-20'), dt64('7699')), 'zkuW'),
                    ),
                ),
                (
                    3,
                    (
                        ((dt64('2210-12-26'), dt64('164167')), 'z2Oo'),
                        ((dt64('2224-04-06'), dt64('43127')), 'z5l6'),
                        ((dt64('2202-08-20'), dt64('7699')), 'zCE3'),
                    ),
                ),
            ),
        )

    def test_frame_set_index_hierarchy_i(self) -> None:
        IACF = IndexAutoConstructorFactory
        f1 = ff.parse('s(3,4)|v(dtD, dtY, str, str)')
        f2 = f1.set_index_hierarchy(
            (0, 1),
            index_constructors=(IACF('a'), IACF),
            drop=True,
        )
        self.assertEqual(f2.index.name, ('a', 1))

        f3 = f1.set_index_hierarchy(
            (0, 1),
            index_constructors=(IACF, IACF('b')),
            drop=True,
        )
        self.assertEqual(f3.index.name, (0, 'b'))

    # ---------------------------------------------------------------------------

    def test_frame_iloc_in_loc_a(self) -> None:
        records = (
            (2, 2, 'a', False, False),
            (30, 34, 'b', True, False),
            (2, 95, 'c', False, False),
            (30, 73, 'd', True, True),
        )
        f1 = Frame.from_records(
            records, columns=('p', 'q', 'r', 's', 't'), index=('w', 'x', 'y', 'z')
        )

        self.assertEqual(
            f1.loc[ILoc[-2:], ['q', 't']].to_pairs(),
            (('q', (('y', 95), ('z', 73))), ('t', (('y', False), ('z', True)))),
        )

        self.assertEqual(
            f1.loc[ILoc[[0, -1]], 's':].to_pairs(),  # type: ignore  # https://github.com/python/typeshed/pull/3024
            (('s', (('w', False), ('z', True))), ('t', (('w', False), ('z', True)))),
        )

        self.assertEqual(
            f1.loc[['w', 'x'], ILoc[[0, -1]]].to_pairs(),
            (('p', (('w', 2), ('x', 30))), ('t', (('w', False), ('x', False)))),
        )

    # ---------------------------------------------------------------------------

    def test_frame_drop_a(self) -> None:
        records = (
            (2, 2, 'a', False, False),
            (30, 34, 'b', True, False),
            (2, 95, 'c', False, False),
            (30, 73, 'd', True, True),
        )
        f1 = Frame.from_records(
            records, columns=('p', 'q', 'r', 's', 't'), index=('w', 'x', 'y', 'z')
        )

        self.assertEqual(
            f1.drop['r':].to_pairs(),  # type: ignore  # https://github.com/python/typeshed/pull/3024
            (
                ('p', (('w', 2), ('x', 30), ('y', 2), ('z', 30))),
                ('q', (('w', 2), ('x', 34), ('y', 95), ('z', 73))),
            ),
        )

        self.assertEqual(
            f1.drop.loc[['x', 'z'], 's':].to_pairs(),  # type: ignore  # https://github.com/python/typeshed/pull/3024
            (
                ('p', (('w', 2), ('y', 2))),
                ('q', (('w', 2), ('y', 95))),
                ('r', (('w', 'a'), ('y', 'c'))),
            ),
        )

        self.assertEqual(
            f1.drop.loc['x':, 'q':].to_pairs(),  # type: ignore  # https://github.com/python/typeshed/pull/3024
            (('p', (('w', 2),)),),
        )

    def test_frame_drop_b(self) -> None:
        records = (
            (2, 2, 'a', False, False),
            (30, 34, 'b', True, False),
        )
        f1 = Frame.from_records(
            records, columns=('p', 'q', 'r', 's', 't'), index=('w', 'x')
        )
        self.assertEqual(
            f1.drop.iloc[1].to_pairs(),
            (
                ('p', (('w', 2),)),
                ('q', (('w', 2),)),
                ('r', (('w', 'a'),)),
                ('s', (('w', False),)),
                ('t', (('w', False),)),
            ),
        )

    def test_frame_drop_c(self) -> None:
        index = IndexHierarchy.from_product(['x'], ['a', 'b'])
        f1 = Frame.from_elements([1, 2], index=index, columns=['a'])
        f2 = f1.drop['a']
        self.assertEqual(f2.shape, (2, 0))

    def test_frame_drop_d(self) -> None:
        columns = sf.IndexHierarchy.from_product([10, 20], ['a', 'b'])

        f1 = Frame(np.arange(8).reshape(2, 4), columns=columns)
        f2 = f1.drop[(20, 'a')]
        self.assertEqual(
            f2.to_pairs(),
            (
                ((10, 'a'), ((0, 0), (1, 4))),
                ((10, 'b'), ((0, 1), (1, 5))),
                ((20, 'b'), ((0, 3), (1, 7))),
            ),
        )

        f3 = f1.drop[(10, 'b') :]  # type: ignore
        self.assertEqual(f3.to_pairs(), (((10, 'a'), ((0, 0), (1, 4))),))

        f4 = f1.drop[[(10, 'b'), (20, 'b')]]
        self.assertEqual(
            f4.to_pairs(),
            (((10, 'a'), ((0, 0), (1, 4))), ((20, 'a'), ((0, 2), (1, 6)))),
        )

        f5 = f1.drop[:]
        self.assertEqual(f5.shape, (2, 0))

        col1_dtype = str(f1.columns.dtypes.values[0])  # type: ignore
        col2_dtype = str(f1.columns.dtypes.values[1])  # type: ignore

        # Check that we can represent the IndexHierarchy
        d = f5.display(DisplayConfig(type_color=False))
        self.assertEqual(
            tuple(d),
            (
                ['<Frame>'],
                ['<IndexHierarchy>', f'<{col1_dtype}>'],
                ['', f'<{col2_dtype}>'],
                ['<Index>'],
                ['0'],
                ['1'],
                ['<int64>'],
            ),
        )

    # ---------------------------------------------------------------------------

    def test_frame_roll_a(self) -> None:
        records = (
            (2, 2, 'a', False, False),
            (30, 34, 'b', True, False),
            (2, 95, 'c', False, False),
            (30, 73, 'd', True, True),
        )
        f1 = Frame.from_records(
            records, columns=('p', 'q', 'r', 's', 't'), index=('w', 'x', 'y', 'z')
        )

        self.assertEqual(
            f1.roll(1).to_pairs(),
            (
                ('p', (('w', 30), ('x', 2), ('y', 30), ('z', 2))),
                ('q', (('w', 73), ('x', 2), ('y', 34), ('z', 95))),
                ('r', (('w', 'd'), ('x', 'a'), ('y', 'b'), ('z', 'c'))),
                ('s', (('w', True), ('x', False), ('y', True), ('z', False))),
                ('t', (('w', True), ('x', False), ('y', False), ('z', False))),
            ),
        )

        self.assertEqual(
            f1.roll(-2, include_index=True).to_pairs(),
            (
                ('p', (('y', 2), ('z', 30), ('w', 2), ('x', 30))),
                ('q', (('y', 95), ('z', 73), ('w', 2), ('x', 34))),
                ('r', (('y', 'c'), ('z', 'd'), ('w', 'a'), ('x', 'b'))),
                ('s', (('y', False), ('z', True), ('w', False), ('x', True))),
                ('t', (('y', False), ('z', True), ('w', False), ('x', False))),
            ),
        )

        self.assertEqual(
            f1.roll(-3, 3).to_pairs(),
            (
                ('p', (('w', 'd'), ('x', 'a'), ('y', 'b'), ('z', 'c'))),
                ('q', (('w', True), ('x', False), ('y', True), ('z', False))),
                ('r', (('w', True), ('x', False), ('y', False), ('z', False))),
                ('s', (('w', 30), ('x', 2), ('y', 30), ('z', 2))),
                ('t', (('w', 73), ('x', 2), ('y', 34), ('z', 95))),
            ),
        )

        self.assertEqual(
            f1.roll(-3, 3, include_index=True, include_columns=True).to_pairs(),
            (
                ('r', (('z', 'd'), ('w', 'a'), ('x', 'b'), ('y', 'c'))),
                ('s', (('z', True), ('w', False), ('x', True), ('y', False))),
                ('t', (('z', True), ('w', False), ('x', False), ('y', False))),
                ('p', (('z', 30), ('w', 2), ('x', 30), ('y', 2))),
                ('q', (('z', 73), ('w', 2), ('x', 34), ('y', 95))),
            ),
        )

    # ---------------------------------------------------------------------------

    def test_frame_shift_a(self) -> None:
        records = (
            (2, 2, 'a', False, False),
            (30, 34, 'b', True, False),
            (2, 95, 'c', False, False),
            (30, 73, 'd', True, True),
        )
        f1 = Frame.from_records(
            records, columns=('p', 'q', 'r', 's', 't'), index=('w', 'x', 'y', 'z')
        )

        dtype = np.dtype

        # nan as default forces floats and objects
        self.assertEqual(
            f1.shift(2).dtypes.values.tolist(),
            [dtype('float64'), dtype('float64'), dtype('O'), dtype('O'), dtype('O')],
        )

        self.assertEqual(
            f1.shift(1, fill_value=-1).to_pairs(),
            (
                ('p', (('w', -1), ('x', 2), ('y', 30), ('z', 2))),
                ('q', (('w', -1), ('x', 2), ('y', 34), ('z', 95))),
                ('r', (('w', -1), ('x', 'a'), ('y', 'b'), ('z', 'c'))),
                ('s', (('w', -1), ('x', False), ('y', True), ('z', False))),
                ('t', (('w', -1), ('x', False), ('y', False), ('z', False))),
            ),
        )

        self.assertEqual(
            f1.shift(1, 1, fill_value=-1).to_pairs(),
            (
                ('p', (('w', -1), ('x', -1), ('y', -1), ('z', -1))),
                ('q', (('w', -1), ('x', 2), ('y', 30), ('z', 2))),
                ('r', (('w', -1), ('x', 2), ('y', 34), ('z', 95))),
                ('s', (('w', -1), ('x', 'a'), ('y', 'b'), ('z', 'c'))),
                ('t', (('w', -1), ('x', False), ('y', True), ('z', False))),
            ),
        )

        self.assertEqual(
            f1.shift(0, 5, fill_value=-1).to_pairs(),
            (
                ('p', (('w', -1), ('x', -1), ('y', -1), ('z', -1))),
                ('q', (('w', -1), ('x', -1), ('y', -1), ('z', -1))),
                ('r', (('w', -1), ('x', -1), ('y', -1), ('z', -1))),
                ('s', (('w', -1), ('x', -1), ('y', -1), ('z', -1))),
                ('t', (('w', -1), ('x', -1), ('y', -1), ('z', -1))),
            ),
        )

    def test_frame_shift_b(self) -> None:
        records = (
            (2, 2, 'a', False, False),
            (30, 34, 'b', True, False),
            (2, 95, 'c', False, False),
        )
        f1 = Frame.from_records(
            records,
            columns=('p', 'q', 'r', 's', 't'),
            index=('x', 'y', 'z'),
            name='test',
        )

        self.assertEqual(
            f1.shift(2, fill_value=FillValueAuto).to_pairs(),
            (
                ('p', (('x', 0), ('y', 0), ('z', 2))),
                ('q', (('x', 0), ('y', 0), ('z', 2))),
                ('r', (('x', ''), ('y', ''), ('z', 'a'))),
                ('s', (('x', False), ('y', False), ('z', False))),
                ('t', (('x', False), ('y', False), ('z', False))),
            ),
        )

        self.assertEqual(
            f1.shift(2, fill_value=FillValueAuto(i=-1, U='na', b=True)).to_pairs(),
            (
                ('p', (('x', -1), ('y', -1), ('z', 2))),
                ('q', (('x', -1), ('y', -1), ('z', 2))),
                ('r', (('x', 'na'), ('y', 'na'), ('z', 'a'))),
                ('s', (('x', True), ('y', True), ('z', False))),
                ('t', (('x', True), ('y', True), ('z', False))),
            ),
        )

    def test_frame_shift_c1(self) -> None:
        f1 = sf.Frame.from_element(1, columns=['a'], index=[1])

        f2 = f1.shift(index=-1, fill_value=0)
        self.assertEqual(f2.to_pairs(), (('a', ((1, 0),)),))

        f3 = f1.shift(columns=-1, fill_value=0)
        self.assertEqual(f3.to_pairs(), (('a', ((1, 0),)),))

    def test_frame_shift_c2(self) -> None:
        f1 = sf.Frame.from_element(1, columns=['a'], index=[1])

        f2 = f1.shift(index=-1, fill_value=[-1])
        self.assertEqual(f2.to_pairs(), (('a', ((1, -1),)),))

        f3 = f1.shift(columns=-1, fill_value=[-1])
        self.assertEqual(f3.to_pairs(), (('a', ((1, -1),)),))

    def test_frame_shift_d1(self) -> None:
        f1 = sf.Frame.from_element(1, columns=['a', 'b', 'c'], index=[1])
        f2 = f1.shift(index=-3, fill_value=0)
        self.assertEqual(
            f2.to_pairs(), (('a', ((1, 0),)), ('b', ((1, 0),)), ('c', ((1, 0),)))
        )

        f3 = f1.shift(index=-5, fill_value=0)
        self.assertEqual(
            f3.to_pairs(), (('a', ((1, 0),)), ('b', ((1, 0),)), ('c', ((1, 0),)))
        )

        f4 = f1.shift(index=3, fill_value=0)
        self.assertEqual(
            f4.to_pairs(), (('a', ((1, 0),)), ('b', ((1, 0),)), ('c', ((1, 0),)))
        )

        f5 = f1.shift(index=5, fill_value=0)
        self.assertEqual(
            f5.to_pairs(), (('a', ((1, 0),)), ('b', ((1, 0),)), ('c', ((1, 0),)))
        )

    def test_frame_shift_d2(self) -> None:
        f1 = sf.Frame.from_element(1, columns=['a', 'b', 'c'], index=[1])
        f2 = f1.shift(index=-3, fill_value=[-1, -2, -3])
        self.assertEqual(
            f2.to_pairs(), (('a', ((1, -1),)), ('b', ((1, -2),)), ('c', ((1, -3),)))
        )

        f3 = f1.shift(index=-5, fill_value=[-1, -2, -3])
        self.assertEqual(
            f3.to_pairs(), (('a', ((1, -1),)), ('b', ((1, -2),)), ('c', ((1, -3),)))
        )

        f4 = f1.shift(index=3, fill_value=[-1, -2, -3])
        self.assertEqual(
            f4.to_pairs(), (('a', ((1, -1),)), ('b', ((1, -2),)), ('c', ((1, -3),)))
        )

        f5 = f1.shift(index=5, fill_value=[-1, -2, -3])
        self.assertEqual(
            f5.to_pairs(), (('a', ((1, -1),)), ('b', ((1, -2),)), ('c', ((1, -3),)))
        )

    # ---------------------------------------------------------------------------

    def test_frame_name_a(self) -> None:
        records = (
            (2, 2, 'a', False, False),
            (30, 34, 'b', True, False),
            (2, 95, 'c', False, False),
        )
        f1 = Frame.from_records(
            records,
            columns=('p', 'q', 'r', 's', 't'),
            index=('x', 'y', 'z'),
            name='test',
        )

        self.assertEqual(f1.name, 'test')

        f2 = f1.rename('alt')

        self.assertEqual(f1.name, 'test')
        self.assertEqual(f2.name, 'alt')

    def test_frame_name_b(self) -> None:
        with self.assertRaises(TypeError):
            f = Frame.from_dict(dict(a=(1, 2), b=(3, 4)), name=['test'])

        with self.assertRaises(TypeError):
            f = Frame.from_dict(dict(a=(1, 2), b=(3, 4)), name={'a': 30})

        with self.assertRaises(TypeError):
            f = Frame.from_dict(dict(a=(1, 2), b=(3, 4)), name=('a', [1, 2]))

    def test_frame_name_c(self) -> None:
        records = (
            (2, 2, 'a', False, False),
            (30, 34, 'b', True, False),
            (2, 95, 'c', False, False),
        )
        f1 = FrameGO.from_records(
            records,
            columns=('p', 'q', 'r', 's', 't'),
            index=('x', 'y', 'z'),
            name='test',
        )

        self.assertEqual(f1.name, 'test')

        f2 = f1.rename('alt')

        self.assertEqual(f1.name, 'test')
        self.assertEqual(f2.name, 'alt')

        f2['u'] = -1  # type: ignore

        self.assertEqual(f1.columns.values.tolist(), ['p', 'q', 'r', 's', 't'])
        self.assertEqual(f2.columns.values.tolist(), ['p', 'q', 'r', 's', 't', 'u'])

    @skip_win
    def test_frame_display_a(self) -> None:
        f1 = Frame.from_records(
            ((1, 2), (True, False)),
            name='foo',
            index=Index(('x', 'y'), name='bar'),
            columns=Index(('a', 'b'), name='rig'),
        )

        d1 = f1.display(
            DisplayConfig(type_color=False, type_show=False, include_columns=False)
        )

        match = tuple(f1.display(DisplayConfig(type_color=False)))

        self.assertEqual(
            match,
            (
                ['<Frame: foo>'],
                ['<Index: rig>', 'a', 'b', '<<U1>'],
                ['<Index: bar>', '', ''],
                ['x', '1', '2'],
                ['y', '1', '0'],
                ['<<U1>', '<int64>', '<int64>'],
            ),
        )

    def test_frame_reindex_drop_level_a(self) -> None:
        f1 = Frame.from_dict_records(
            (dict(a=x, b=x) for x in range(4)),
            index=sf.IndexHierarchy.from_labels([(1, 1), (1, 2), (2, 3), (2, 4)]),
        )

        with self.assertRaises(Exception):
            # this results in an index of size 2 being created, as we dro the leves with a postive depth; next support negative depth?
            f2 = f1.relabel_level_drop(index=-1)

    # ---------------------------------------------------------------------------

    def test_frame_clip_a(self) -> None:
        records = (
            (2, 2),
            (30, 34),
            (2, 95),
        )
        f1 = Frame.from_records(records, columns=('a', 'b'), index=('x', 'y', 'z'))

        self.assertEqual(
            f1.clip(upper=0).to_pairs(),
            (
                ('a', (('x', 0), ('y', 0), ('z', 0))),
                ('b', (('x', 0), ('y', 0), ('z', 0))),
            ),
        )

        self.assertEqual(
            f1.clip(lower=90).to_pairs(),
            (
                ('a', (('x', 90), ('y', 90), ('z', 90))),
                ('b', (('x', 90), ('y', 90), ('z', 95))),
            ),
        )

    def test_frame_clip_b(self) -> None:
        records = (
            (2, 2),
            (30, 34),
            (2, 95),
        )
        f1 = Frame.from_records(records, columns=('a', 'b'), index=('x', 'y', 'z'))

        s1 = Series((1, 20), index=('a', 'b'))

        self.assertEqual(
            f1.clip(upper=s1, axis=1).to_pairs(),
            (
                ('a', (('x', 1), ('y', 1), ('z', 1))),
                ('b', (('x', 2), ('y', 20), ('z', 20))),
            ),
        )

        s2 = Series((3, 33, 80), index=('x', 'y', 'z'))

        self.assertEqual(
            f1.clip(lower=s2, axis=0).to_pairs(),
            (
                ('a', (('x', 3), ('y', 33), ('z', 80))),
                ('b', (('x', 3), ('y', 34), ('z', 95))),
            ),
        )

    def test_frame_clip_c(self) -> None:
        records = (
            (2, 2),
            (30, 34),
            (2, 95),
        )
        f1 = Frame.from_records(records, columns=('a', 'b'), index=('x', 'y', 'z'))

        f2 = sf.Frame.from_records(
            [[5, 4], [0, 10]], index=list('yz'), columns=list('ab')
        )

        self.assertEqual(
            f1.clip(upper=f2).to_pairs(),
            (
                ('a', (('x', 2.0), ('y', 5.0), ('z', 0.0))),
                ('b', (('x', 2.0), ('y', 4.0), ('z', 10.0))),
            ),
        )

    @unittest.skip('precedence of min/max changed in numpy 1.17.4')
    def test_frame_clip_d(self) -> None:
        records = (
            (2, 2),
            (30, 34),
            (2, 95),
        )
        f1 = Frame.from_records(records, columns=('a', 'b'), index=('x', 'y', 'z'))

        f2 = sf.Frame([[5, 4], [0, 10]], index=list('yz'), columns=list('ab'))

        self.assertEqual(
            f1.clip(lower=3, upper=f2).to_pairs(),
            (
                ('a', (('x', 3.0), ('y', 5.0), ('z', 3.0))),
                ('b', (('x', 3.0), ('y', 4.0), ('z', 10.0))),
            ),
        )

    def test_frame_clip_e(self) -> None:
        records = (
            (2, 2),
            (30, 34),
            (22, 95),
        )
        f1 = Frame.from_records(records, columns=('a', 'b'), index=('x', 'y', 'z'))
        f2 = f1.clip(lower=20, upper=31)
        self.assertEqual(
            f2.to_pairs(),
            (
                ('a', (('x', 20), ('y', 30), ('z', 22))),
                ('b', (('x', 20), ('y', 31), ('z', 31))),
            ),
        )

    def test_frame_clip_f(self) -> None:
        records = (
            (2, 2),
            (30, 34),
            (22, 95),
        )
        f1 = Frame.from_records(records, columns=('a', 'b'), index=('x', 'y', 'z'))
        f2 = f1.clip()

        self.assertEqual(
            f2.to_pairs(),
            (
                ('a', (('x', 2), ('y', 30), ('z', 22))),
                ('b', (('x', 2), ('y', 34), ('z', 95))),
            ),
        )

    def test_frame_clip_g(self) -> None:
        records = (
            (2, 2),
            (30, 34),
            (22, 95),
        )
        f1 = Frame.from_records(records, columns=('a', 'b'), index=('x', 'y', 'z'))

        with self.assertRaises(RuntimeError):
            _ = f1.clip(upper=Series((1, 10), index=('b', 'c')))

        with self.assertRaises(RuntimeError):
            _ = f1.clip(upper=(3, 4))

        f2 = f1.clip(upper=Series((1, 10), index=('b', 'c')), axis=1)
        self.assertEqual(
            f2.to_pairs(),
            (
                ('a', (('x', 2.0), ('y', 30.0), ('z', 22.0))),
                ('b', (('x', 1.0), ('y', 1.0), ('z', 1.0))),
            ),
        )

    def test_frame_clip_h(self) -> None:
        dt64 = np.datetime64
        records = (
            (2, dt64('2020')),
            (30, dt64('1995')),
            (22, dt64('1980')),
        )
        f1 = Frame.from_records(records, columns=('a', 'b'), index=('x', 'y', 'z'))

        f2 = f1.clip(lower=Series((10, dt64('2001')), index=('a', 'b')), axis=1)
        self.assertEqual([d.kind for d in f2.dtypes.values], ['i', 'M'])

        self.assertEqual(
            f2.to_pairs(),
            (
                ('a', (('x', 10), ('y', 30), ('z', 22))),
                ('b', (('x', dt64('2020')), ('y', dt64('2001')), ('z', dt64('2001')))),
            ),
        )

    def test_frame_clip_i(self) -> None:
        dt64 = np.datetime64
        records = (
            (2, dt64('2020')),
            (30, dt64('1995')),
            (22, dt64('1980')),
        )
        f1 = Frame.from_records(records, columns=('a', 'b'), index=('x', 'y', 'z'))

        lb = Frame.from_fields(
            ((25, 25, 25), (dt64('2001'), dt64('2001'), dt64('2001'))),
            columns=('a', 'b'),
            index=('x', 'y', 'z'),
        )

        f2 = f1.clip(lower=lb, axis=1)

        self.assertEqual(
            f2.to_pairs(),
            (
                ('a', (('x', 25), ('y', 30), ('z', 25))),
                (
                    'b',
                    (
                        ('x', np.datetime64('2020')),
                        ('y', np.datetime64('2001')),
                        ('z', np.datetime64('2001')),
                    ),
                ),
            ),
        )

    def test_frame_clip_j(self) -> None:
        f1 = Frame.from_fields(
            ((10, 20, 30), (40, 20, 30), (False, True, False), (True, False, True)),
            columns=('a', 'b', 'c', 'd'),
            index=('x', 'y', 'z'),
            consolidate_blocks=True,
        )
        f2 = f1.clip(lower=f1 * 2)
        self.assertEqual([d.kind for d in f2.dtypes.values], ['i', 'i', 'i', 'i'])
        self.assertEqual(
            f2.to_pairs(),
            (
                ('a', (('x', 20), ('y', 40), ('z', 60))),
                ('b', (('x', 80), ('y', 40), ('z', 60))),
                ('c', (('x', 0), ('y', 2), ('z', 0))),
                ('d', (('x', 2), ('y', 0), ('z', 2))),
            ),
        )

    def test_frame_clip_k(self) -> None:
        f1 = Frame.from_fields(
            ((10, 20, 30), (40, 20, 30), (False, True, False), (True, False, True)),
            columns=('a', 'b', 'c', 'd'),
            index=('x', 'y', 'z'),
        )
        lb = Frame.from_element(20, columns=f1.columns, index=f1.index)
        f2 = f1.clip(lower=lb)

        self.assertEqual(
            f2.to_pairs(),
            (
                ('a', (('x', 20), ('y', 20), ('z', 30))),
                ('b', (('x', 40), ('y', 20), ('z', 30))),
                ('c', (('x', 20), ('y', 20), ('z', 20))),
                ('d', (('x', 20), ('y', 20), ('z', 20))),
            ),
        )

    def test_frame_clip_l(self) -> None:
        f1 = ff.parse('s(2,6)|v(int,int,int,float,float,float)')
        f2 = ff.parse('s(2,6)|v(int,float)') * 0
        f3 = f1.clip(lower=f2)
        self.assertEqual(
            round(f3).to_pairs(),
            (
                (0, ((0, 0.0), (1, 92867.0))),
                (1, ((0, 162197.0), (1, 0.0))),
                (2, ((0, 0.0), (1, 91301.0))),
                (3, ((0, 1080.0), (1, 2580.0))),
                (4, ((0, 3512.0), (1, 1175.0))),
                (5, ((0, 1857.0), (1, 1699.0))),
            ),
        )

    # ---------------------------------------------------------------------------

    def test_frame_from_dict_a(self) -> None:
        with self.assertRaises(RuntimeError):
            # mismatched length
            sf.Frame.from_dict(dict(a=(1, 2, 3, 4, 5), b=tuple('abcdef')))

    def test_frame_from_dict_b(self) -> None:
        f = Frame.from_dict(
            {('a', 1): (1, 2), ('a', 2): (3, 4)},
            columns_constructor=IndexHierarchy.from_labels,
        )

        self.assertEqual(f.columns.__class__, IndexHierarchy)
        self.assertEqual(
            f.to_pairs(), ((('a', 1), ((0, 1), (1, 2))), (('a', 2), ((0, 3), (1, 4))))
        )

    def test_frame_from_dict_c(self) -> None:
        a = Frame.from_dict(
            dict(a=(1, 2, 3, 4), b=(5, 6, 7, 8)),
            index=tuple('wxyz'),
            index_constructor=IndexDefaultConstructorFactory('foo'),
            columns_constructor=IndexDefaultConstructorFactory('bar'),
        )
        self.assertEqual(a.index.name, 'foo')
        self.assertEqual(a.columns.name, 'bar')

    # ---------------------------------------------------------------------------

    def test_frame_from_sql_a1(self) -> None:
        conn: sqlite3.Connection = self.get_test_db_e()

        f1 = sf.Frame.from_sql(
            'select * from events', connection=conn, dtypes={'date': 'datetime64[D]'}
        )
        self.assertEqual([dt.kind for dt in f1.dtypes.values], ['M', 'U', 'f', 'i'])

        f2 = sf.Frame.from_sql(
            'select * from events',
            connection=conn,
            dtypes={'date': 'datetime64[D]'},
            index_depth=2,
            index_constructors=(IndexDate, Index),
        )

        self.assertEqual(
            [dt.kind for dt in f2.index.dtypes.values],  # type: ignore
            ['M', 'U'],
        )

        self.assertEqual(
            f2.to_pairs(),
            (
                (
                    'value',
                    (
                        ((np.datetime64('2006-01-01'), 'a1'), 12.5),
                        ((np.datetime64('2006-01-01'), 'b2'), 12.5),
                        ((np.datetime64('2006-01-02'), 'a1'), 12.5),
                        ((np.datetime64('2006-01-02'), 'b2'), 12.5),
                    ),
                ),
                (
                    'count',
                    (
                        ((np.datetime64('2006-01-01'), 'a1'), 20),
                        ((np.datetime64('2006-01-01'), 'b2'), 21),
                        ((np.datetime64('2006-01-02'), 'a1'), 22),
                        ((np.datetime64('2006-01-02'), 'b2'), 23),
                    ),
                ),
            ),
        )

    def test_frame_from_sql_a2(self) -> None:
        conn: sqlite3.Connection = self.get_test_db_e()

        f1 = sf.Frame.from_sql(
            'select * from events where identifier = "b"',
            connection=conn,
            dtypes={'date': 'datetime64[D]'},
        )
        self.assertEqual(f1.shape, (0, 4))

    def test_frame_from_sql_b(self) -> None:
        conn: sqlite3.Connection = self.get_test_db_f()

        f1 = sf.Frame.from_sql(
            'select * from events',
            connection=conn,
            dtypes={'date': 'datetime64[D]', 'count': 'float'},
            index_depth=1,
        )
        self.assertEqual(f1.index.dtype.kind, 'f')  # type: ignore
        self.assertEqual(
            f1.to_pairs(),
            (
                (
                    'date',
                    (
                        (20.0, np.datetime64('2006-01-01')),
                        (21.0, np.datetime64('2006-01-01')),
                        (22.0, np.datetime64('2006-01-02')),
                        (23.0, np.datetime64('2006-01-02')),
                    ),
                ),
                (
                    'identifier',
                    ((20.0, 'a1'), (21.0, 'b2'), (22.0, 'a1'), (23.0, 'b2')),
                ),
                ('value', ((20.0, 12.5), (21.0, 12.5), (22.0, 12.5), (23.0, 12.5))),
            ),
        )

    def test_frame_from_sql_c(self) -> None:
        conn: sqlite3.Connection = self.get_test_db_e()

        f1 = sf.Frame.from_sql(
            'select * from events where identifier=?',
            connection=conn,
            dtypes={'date': 'datetime64[D]'},
            parameters=('a1',),
        )
        self.assertEqual([dt.kind for dt in f1.dtypes.values], ['M', 'U', 'f', 'i'])
        self.assertEqual(
            f1.to_pairs(),
            (
                (
                    'date',
                    (
                        (0, np.datetime64('2006-01-01')),
                        (1, np.datetime64('2006-01-02')),
                    ),
                ),
                ('identifier', ((0, 'a1'), (1, 'a1'))),
                ('value', ((0, 12.5), (1, 12.5))),
                ('count', ((0, 20), (1, 22))),
            ),
        )

    def test_frame_from_sql_d(self) -> None:
        conn: sqlite3.Connection = self.get_test_db_e()

        f1 = sf.Frame.from_sql(
            'select * from events',
            connection=conn,
            index_depth=2,
            index_constructors=(IndexDate, Index),
        )
        self.assertEqual(f1.index.depth, 2)
        self.assertEqual(f1.index.index_types.values.tolist(), [IndexDate, Index])

    # ---------------------------------------------------------------------------

    def test_frame_from_sql_no_args(self) -> None:
        conn: sqlite3.Connection = self.get_test_db_a()

        f1 = sf.Frame.from_sql('select * from events', connection=conn)

        # this might be different on windows
        self.assertEqual([x.kind for x in f1.dtypes.values], ['U', 'U', 'f', 'i'])

        self.assertEqual(
            f1.to_pairs(),
            (
                (
                    'date',
                    (
                        (0, '2006-01-01'),
                        (1, '2006-01-02'),
                        (2, '2006-01-01'),
                        (3, '2006-01-02'),
                    ),
                ),
                ('identifier', ((0, 'a1'), (1, 'a1'), (2, 'b2'), (3, 'b2'))),
                ('value', ((0, 12.5), (1, 12.5), (2, 12.5), (3, 12.5))),
                ('count', ((0, 8), (1, 8), (2, 8), (3, 8))),
            ),
        )

    def test_frame_from_sql_columns_select_no_columns(self) -> None:
        conn: sqlite3.Connection = self.get_test_db_a()

        f1 = sf.Frame.from_sql(
            'select * from events',
            connection=conn,
            columns_depth=0,
        )

        f2 = sf.Frame.from_sql(
            'select * from events',
            connection=conn,
            columns_depth=0,
            columns_select=['date', 'value', 'count'],
        )

        f3 = sf.Frame.from_sql(
            'select * from events',
            connection=conn,
            columns_depth=0,
            columns_select=['count'],
        )

        self.assertEqual([x.kind for x in f1.dtypes.values], ['U', 'U', 'f', 'i'])
        self.assertEqual([x.kind for x in f2.dtypes.values], ['U', 'f', 'i'])
        self.assertEqual([x.kind for x in f3.dtypes.values], ['i'])

        self.assertEqual(
            f1.to_pairs(),
            (
                (
                    0,
                    (
                        (0, '2006-01-01'),
                        (1, '2006-01-02'),
                        (2, '2006-01-01'),
                        (3, '2006-01-02'),
                    ),
                ),
                (1, ((0, 'a1'), (1, 'a1'), (2, 'b2'), (3, 'b2'))),
                (2, ((0, 12.5), (1, 12.5), (2, 12.5), (3, 12.5))),
                (3, ((0, 8), (1, 8), (2, 8), (3, 8))),
            ),
        )
        self.assertEqual(
            f2.to_pairs(),
            (
                (
                    0,
                    (
                        (0, '2006-01-01'),
                        (1, '2006-01-02'),
                        (2, '2006-01-01'),
                        (3, '2006-01-02'),
                    ),
                ),
                (1, ((0, 12.5), (1, 12.5), (2, 12.5), (3, 12.5))),
                (2, ((0, 8), (1, 8), (2, 8), (3, 8))),
            ),
        )
        self.assertEqual(f3.to_pairs(), ((0, ((0, 8), (1, 8), (2, 8), (3, 8))),))

    def test_frame_from_sql_columns_select(self) -> None:
        conn: sqlite3.Connection = self.get_test_db_a()

        f1 = sf.Frame.from_sql(
            'select * from events',
            connection=conn,
            columns_select=['date', 'value', 'count'],
        )

        # this might be different on windows
        self.assertEqual([x.kind for x in f1.dtypes.values], ['U', 'f', 'i'])

        self.assertEqual(
            f1.to_pairs(),
            (
                (
                    'date',
                    (
                        (0, '2006-01-01'),
                        (1, '2006-01-02'),
                        (2, '2006-01-01'),
                        (3, '2006-01-02'),
                    ),
                ),
                ('value', ((0, 12.5), (1, 12.5), (2, 12.5), (3, 12.5))),
                ('count', ((0, 8), (1, 8), (2, 8), (3, 8))),
            ),
        )

    def test_frame_from_sql_columns_select_w_idx(self) -> None:
        conn: sqlite3.Connection = self.get_test_db_b()

        f1 = sf.Frame.from_sql(
            'select * from events',
            connection=conn,
            index_depth=1,
            columns_select=['date', 'value', 'count'],
        )

        # this might be different on windows
        self.assertEqual([x.kind for x in f1.dtypes.values], ['U', 'f', 'i'])

        self.assertEqual(
            f1.to_pairs(),
            (
                (
                    'date',
                    (
                        (0, '2006-01-01'),
                        (1, '2006-01-02'),
                        (2, '2006-01-01'),
                        (3, '2006-01-02'),
                    ),
                ),
                ('value', ((0, 12.5), (1, 12.5), (2, 12.5), (3, 12.5))),
                ('count', ((0, 8), (1, 8), (2, 8), (3, 8))),
            ),
        )

    def test_frame_from_sql_columns_select_w_idx_h(self) -> None:
        conn: sqlite3.Connection = self.get_test_db_b()

        f1 = sf.Frame.from_sql(
            'select * from events',
            connection=conn,
            index_depth=3,
            columns_select=['value', 'count'],
        )

        # this might be different on windows
        self.assertEqual([x.kind for x in f1.dtypes.values], ['f', 'i'])

        self.assertEqual(
            f1.to_pairs(),
            (
                (
                    'value',
                    (
                        ((0, '2006-01-01', 'a1'), 12.5),
                        ((1, '2006-01-02', 'a1'), 12.5),
                        ((2, '2006-01-01', 'b2'), 12.5),
                        ((3, '2006-01-02', 'b2'), 12.5),
                    ),
                ),
                (
                    'count',
                    (
                        ((0, '2006-01-01', 'a1'), 8),
                        ((1, '2006-01-02', 'a1'), 8),
                        ((2, '2006-01-01', 'b2'), 8),
                        ((3, '2006-01-02', 'b2'), 8),
                    ),
                ),
            ),
        )

    def test_frame_from_sql_columns_select_w_col_h(self) -> None:
        conn: sqlite3.Connection = self.get_test_db_c()

        f1 = sf.Frame.from_sql(
            'select * from events',
            connection=conn,
            columns_depth=2,
            columns_select=[('date', 'to'), ('value', 'a'), ('value', 'b')],
        )

        # this might be different on windows
        self.assertEqual([x.kind for x in f1.dtypes.values], ['U', 'f', 'i'])

        self.assertEqual(
            f1.to_pairs(),
            (
                (('date', 'to'), ((0, 'a1'), (1, 'a1'), (2, 'b2'), (3, 'b2'))),
                (('value', 'a'), ((0, 12.5), (1, 12.5), (2, 12.5), (3, 12.5))),
                (('value', 'b'), ((0, 8), (1, 8), (2, 8), (3, 8))),
            ),
        )

    def test_frame_from_sql_columns_select_w_idx_col_h(self) -> None:
        conn: sqlite3.Connection = self.get_test_db_d()

        f1 = sf.Frame.from_sql(
            'select * from events',
            connection=conn,
            index_depth=2,
            columns_depth=2,
            columns_select=[('date', 'to'), ('value', 'a')],
        )

        # this might be different on windows
        self.assertEqual([x.kind for x in f1.dtypes.values], ['U', 'f'])

        self.assertEqual(
            f1.to_pairs(),
            (
                (
                    ('date', 'to'),
                    (
                        (('0', '2006-01-01'), 'a1'),
                        (('1', '2006-01-02'), 'a1'),
                        (('2', '2006-01-01'), 'b2'),
                        (('3', '2006-01-02'), 'b2'),
                    ),
                ),
                (
                    ('value', 'a'),
                    (
                        (('0', '2006-01-01'), 12.5),
                        (('1', '2006-01-02'), 12.5),
                        (('2', '2006-01-01'), 12.5),
                        (('3', '2006-01-02'), 12.5),
                    ),
                ),
            ),
        )

    # ---------------------------------------------------------------------------

    def test_frame_from_records_items_a(self) -> None:
        def gen() -> tp.Iterator[tp.Tuple[TLabel, tp.Dict[TLabel, tp.Any]]]:
            for i in range(3):
                yield f'000{i}', {'squared': i**2, 'cubed': i**3}

        f = Frame.from_dict_records_items(gen())

        self.assertEqual(
            f.to_pairs(),
            (
                ('squared', (('0000', 0), ('0001', 1), ('0002', 4))),
                ('cubed', (('0000', 0), ('0001', 1), ('0002', 8))),
            ),
        )

    def test_frame_from_records_items_b(self) -> None:
        def gen() -> tp.Iterator[tp.Tuple[TLabel, tp.Tuple[str, str]]]:
            for i in range(3):
                yield f'000{i}', ('a' * i, 'b' * i)

        f = Frame.from_records_items(
            gen(), index_constructor=IndexDefaultConstructorFactory('foo')
        )
        self.assertEqual(
            f.to_pairs(),
            (
                (0, (('0000', ''), ('0001', 'a'), ('0002', 'aa'))),
                (1, (('0000', ''), ('0001', 'b'), ('0002', 'bb'))),
            ),
        )
        self.assertEqual(f.index.name, 'foo')

    # ---------------------------------------------------------------------------

    def test_frame_count_a(self) -> None:
        records = (
            (2, 2),
            (np.nan, 34),
            (2, -95),
        )
        f1 = Frame.from_records(records, columns=('a', 'b'), index=('x', 'y', 'z'))

        self.assertEqual(f1.count(axis=0).to_pairs(), (('a', 2), ('b', 3)))
        self.assertEqual(f1.count(axis=1).to_pairs(), (('x', 2), ('y', 1), ('z', 2)))

        # can reuse index instance on both axis
        self.assertEqual(id(f1.index), id(f1.count(axis=1).index))
        self.assertEqual(id(f1.columns), id(f1.count(axis=0).index))

    def test_frame_count_b(self) -> None:
        records = (
            (2, 2),
            (np.nan, 34),
            (2, -95),
        )
        f1 = FrameGO.from_records(records, columns=('a', 'b'), index=('x', 'y', 'z'))

        self.assertEqual(f1.count(axis=0).to_pairs(), (('a', 2), ('b', 3)))
        self.assertEqual(f1.count(axis=1).to_pairs(), (('x', 2), ('y', 1), ('z', 2)))

        # for axis 1, can reuse index instance from FrameGO
        self.assertEqual(id(f1.index), id(f1.count(axis=1).index))
        # for axis 0, cannot reuse columns instance as is mutable
        self.assertNotEqual(id(f1.columns), id(f1.count(axis=0).index))

    def test_frame_count_c(self) -> None:
        records = (
            (2, 2),
            (np.nan, 34),
            (2, -95),
        )
        f1 = FrameGO.from_records(records, columns=('a', 'b'), index=('x', 'y', 'z'))

        self.assertEqual(f1.count(axis=0, skipna=False).to_pairs(), (('a', 3), ('b', 3)))
        self.assertEqual(
            f1.count(axis=1, skipna=False).to_pairs(), (('x', 2), ('y', 2), ('z', 2))
        )

    def test_frame_count_d(self) -> None:
        records = (
            (2, 2),
            (np.nan, 34),
            (2, 0),
        )
        f1 = FrameGO.from_records(records, columns=('a', 'b'), index=('x', 'y', 'z'))

        self.assertEqual(f1.count(axis=0, unique=True).to_pairs(), (('a', 1), ('b', 3)))
        self.assertEqual(f1.count(axis=0, unique=False).to_pairs(), (('a', 2), ('b', 3)))

        with self.assertRaises(RuntimeError):
            f1.count(axis=0, skipfalsy=True, skipna=False, unique=False)

        self.assertEqual(
            f1.count(axis=0, skipfalsy=True, unique=False).to_pairs(),
            (('a', 2), ('b', 2)),
        )
        self.assertEqual(
            f1.count(axis=0, skipfalsy=True, unique=True).to_pairs(),
            (('a', 1), ('b', 2)),
        )

        self.assertEqual(
            f1.count(axis=1, skipna=False).to_pairs(), (('x', 2), ('y', 2), ('z', 2))
        )
        self.assertEqual(
            f1.count(axis=1, skipfalsy=True).to_pairs(), (('x', 2), ('y', 1), ('z', 1))
        )
        self.assertEqual(
            f1.count(axis=1, skipfalsy=True, unique=True).to_pairs(),
            (('x', 1), ('y', 1), ('z', 1)),
        )
        self.assertEqual(
            f1.count(axis=1, unique=True).to_pairs(), (('x', 1), ('y', 1), ('z', 2))
        )

    def test_frame_count_e(self) -> None:
        records = (
            (2, 2),
            (2, 34),
            (2, -95),
        )
        f1 = FrameGO.from_records(records, columns=('a', 'b'), index=('x', 'y', 'z'))
        f2 = f1.count(axis=0, unique=True, skipna=False, skipfalsy=False)
        self.assertEqual(f2.to_pairs(), (('a', 1), ('b', 3)))

    # ---------------------------------------------------------------------------

    def test_frame_loc_min_a(self) -> None:
        records = (
            (2, 2),
            (30, 34),
            (2, -95),
        )
        f1 = Frame.from_records(records, columns=('a', 'b'), index=('x', 'y', 'z'))

        self.assertEqual(f1.loc_min().to_pairs(), (('a', 'x'), ('b', 'z')))

        self.assertEqual(
            f1.loc_min(axis=1).to_pairs(), (('x', 'a'), ('y', 'a'), ('z', 'b'))
        )

    def test_frame_loc_min_b(self) -> None:
        records = (
            (2, 2),
            (np.nan, 34),
            (2, -95),
        )
        f1 = Frame.from_records(records, columns=('a', 'b'), index=('x', 'y', 'z'))

        with self.assertRaises(RuntimeError):
            f1.loc_min(skipna=False)

    def test_frame_iloc_min_a(self) -> None:
        records = (
            (2, 2),
            (30, 34),
            (2, -95),
        )
        f1 = Frame.from_records(records, columns=('a', 'b'), index=('x', 'y', 'z'))

        self.assertEqual(f1.iloc_min().to_pairs(), (('a', 0), ('b', 2)))

        self.assertEqual(f1.iloc_min(axis=1).to_pairs(), (('x', 0), ('y', 0), ('z', 1)))

    def test_frame_iloc_min_b(self) -> None:
        records = (
            (2, 2),
            (30, np.nan),
            (2, -95),
        )
        f1 = Frame.from_records(records, columns=('a', 'b'), index=('x', 'y', 'z'))

        self.assertAlmostEqualItems(
            f1.iloc_min(skipna=False).to_pairs(), (('a', 0), ('b', np.nan))
        )

        self.assertAlmostEqualItems(
            f1.iloc_min(axis=1, skipna=False).to_pairs(),
            (('x', 0), ('y', np.nan), ('z', 1)),
        )

    def test_frame_loc_max_a(self) -> None:
        records = (
            (2000, 2),
            (30, 34),
            (2, -95),
        )
        f1 = Frame.from_records(records, columns=('a', 'b'), index=('x', 'y', 'z'))

        self.assertEqual(f1.loc_max().to_pairs(), (('a', 'x'), ('b', 'y')))

        self.assertEqual(
            f1.loc_max(axis=1).to_pairs(), (('x', 'a'), ('y', 'b'), ('z', 'a'))
        )

    def test_frame_loc_max_b(self) -> None:
        records = (
            (2, 2),
            (np.nan, 34),
            (2, -95),
        )
        f1 = Frame.from_records(records, columns=('a', 'b'), index=('x', 'y', 'z'))

        with self.assertRaises(RuntimeError):
            f1.loc_max(skipna=False)

    def test_frame_loc_max_c(self) -> None:
        f = Frame.from_records(
            [[1, 4], [2, 3]],
            index=IndexHierarchy.from_labels(
                [['A1', 'a1'], ['B1', 'b1']], name=tuple('xy')
            ),
            columns=tuple('ab'),
        )
        s = f.loc_max()
        expected = Series(
            [('B1', 'b1'), ('A1', 'a1')], index=tuple('ab'), name=tuple('xy')
        )
        assert s.equals(expected)

    def test_frame_loc_max_d(self) -> None:
        f = Frame.from_elements(
            [1, 2], index=IndexHierarchy.from_labels([[1, 2], [3, 4]])
        )
        s = f.loc_max()
        expected = Series(
            [(3, 4)],
            index=[0],
        )
        assert s.equals(expected)

    def test_frame_iloc_max_a(self) -> None:
        records = (
            (2000, 2),
            (30, 34),
            (2, -95),
        )
        f1 = Frame.from_records(records, columns=('a', 'b'), index=('x', 'y', 'z'))

        self.assertEqual(f1.iloc_max().to_pairs(), (('a', 0), ('b', 1)))

        self.assertEqual(f1.iloc_max(axis=1).to_pairs(), (('x', 0), ('y', 1), ('z', 0)))

    # ---------------------------------------------------------------------------

    def test_frame_loc_notna_a(self) -> None:
        records = (
            (np.nan, 2, 5, 4),
            (30, np.nan, np.nan, np.nan),
            (2, np.nan, 3, np.nan),
        )
        f1 = Frame.from_records(
            records, columns=('a', 'b', 'c', 'd'), index=('x', 'y', 'z')
        )

        self.assertEqual(
            f1.loc_notna_first().to_pairs(),
            (('a', 'y'), ('b', 'x'), ('c', 'x'), ('d', 'x')),
        )
        self.assertEqual(
            f1.loc_notna_last().to_pairs(),
            (('a', 'z'), ('b', 'x'), ('c', 'z'), ('d', 'x')),
        )

    def test_frame_loc_notna_b(self) -> None:
        records = (
            (np.nan, 2, 5, 4),
            (30, 1, np.nan, np.nan),
            (2, np.nan, 3, 4),
        )
        f1 = Frame.from_records(
            records, columns=('a', 'b', 'c', 'd'), index=('x', 'y', 'z')
        )

        self.assertEqual(
            f1.loc_notna_first(axis=1).to_pairs(), (('x', 'b'), ('y', 'a'), ('z', 'a'))
        )
        self.assertEqual(
            f1.loc_notna_last(axis=1).to_pairs(), (('x', 'd'), ('y', 'b'), ('z', 'd'))
        )

    def test_frame_loc_notna_c(self) -> None:
        records = (
            (np.nan, 2, 5, 4),
            (np.nan, np.nan, np.nan, np.nan),
            (np.nan, np.nan, 3, 4),
        )
        f1 = Frame.from_records(
            records, columns=('a', 'b', 'c', 'd'), index=('x', 'y', 'z')
        )
        post = f1.loc_notna_first(fill_value=None)
        self.assertEqual(
            post.to_pairs(), (('a', None), ('b', 'x'), ('c', 'x'), ('d', 'x'))
        )

    def test_frame_loc_notna_d(self) -> None:
        records = (
            (None, None, None, None),
            (None, None, None, None),
            (None, None, None, None),
        )
        f1 = Frame.from_records(
            records, columns=('a', 'b', 'c', 'd'), index=('x', 'y', 'z')
        )
        self.assertEqual(
            f1.loc_notna_first(fill_value=None, axis=0).to_pairs(),
            (('a', None), ('b', None), ('c', None), ('d', None)),
        )
        self.assertEqual(
            f1.loc_notna_first(fill_value=None, axis=1).to_pairs(),
            (('x', None), ('y', None), ('z', None)),
        )

    def test_frame_iloc_notna_a(self) -> None:
        records = (
            (np.nan, 2, 5, 4),
            (30, np.nan, np.nan, np.nan),
            (2, np.nan, 3, np.nan),
        )
        f1 = Frame.from_records(
            records, columns=('a', 'b', 'c', 'd'), index=('x', 'y', 'z')
        )

        self.assertEqual(
            f1.iloc_notna_first().to_pairs(), (('a', 1), ('b', 0), ('c', 0), ('d', 0))
        )
        self.assertEqual(
            f1.iloc_notna_last().to_pairs(), (('a', 2), ('b', 0), ('c', 2), ('d', 0))
        )

    def test_frame_iloc_notna_b(self) -> None:
        records = (
            (np.nan, 2, 5, 4),
            (30, 1, np.nan, np.nan),
            (2, np.nan, 3, 4),
        )
        f1 = Frame.from_records(
            records, columns=('a', 'b', 'c', 'd'), index=('x', 'y', 'z')
        )

        self.assertEqual(
            f1.iloc_notna_first(axis=1).to_pairs(), (('x', 1), ('y', 0), ('z', 0))
        )
        self.assertEqual(
            f1.iloc_notna_last(axis=1).to_pairs(), (('x', 3), ('y', 1), ('z', 3))
        )

    def test_frame_iloc_notna_c(self) -> None:
        records = (
            (np.nan, 2, 5, 4),
            (np.nan, np.nan, np.nan, np.nan),
            (np.nan, np.nan, 3, 4),
        )
        f1 = Frame.from_records(
            records,
            columns=IndexHierarchy.from_product((0, 1), (10, 100)),
            index=('x', 'y', 'z'),
        )
        post = f1.loc_notna_first(axis=1, fill_value='')
        self.assertEqual(post.to_pairs(), (('x', (0, 100)), ('y', ''), ('z', (1, 10))))

    # ---------------------------------------------------------------------------

    def test_frame_loc_notfalsy_a(self) -> None:
        records = (
            (0, 2, 5, 4),
            (30, 0, 0, 0),
            (2, 0, 3, 0),
        )
        f1 = Frame.from_records(
            records, columns=('a', 'b', 'c', 'd'), index=('x', 'y', 'z')
        )

        self.assertEqual(
            f1.loc_notfalsy_first().to_pairs(),
            (('a', 'y'), ('b', 'x'), ('c', 'x'), ('d', 'x')),
        )
        self.assertEqual(
            f1.loc_notfalsy_last().to_pairs(),
            (('a', 'z'), ('b', 'x'), ('c', 'z'), ('d', 'x')),
        )

    def test_frame_loc_notfalsy_b(self) -> None:
        records = (
            (0, 2, 5, 4),
            (30, 1, 0, 0),
            (2, 0, 3, 4),
        )
        f1 = Frame.from_records(
            records, columns=('a', 'b', 'c', 'd'), index=('x', 'y', 'z')
        )

        self.assertEqual(
            f1.loc_notfalsy_first(axis=1).to_pairs(),
            (('x', 'b'), ('y', 'a'), ('z', 'a')),
        )
        self.assertEqual(
            f1.loc_notfalsy_last(axis=1).to_pairs(),
            (('x', 'd'), ('y', 'b'), ('z', 'd')),
        )

    def test_frame_iloc_notfalsy_a(self) -> None:
        records = (
            (0, 2, 5, 4),
            (30, 0, 0, 0),
            (2, 0, 3, 0),
        )
        f1 = Frame.from_records(
            records, columns=('a', 'b', 'c', 'd'), index=('x', 'y', 'z')
        )

        self.assertEqual(
            f1.iloc_notfalsy_first().to_pairs(),
            (('a', 1), ('b', 0), ('c', 0), ('d', 0)),
        )
        self.assertEqual(
            f1.iloc_notfalsy_last().to_pairs(), (('a', 2), ('b', 0), ('c', 2), ('d', 0))
        )

    def test_frame_iloc_notfalsy_b(self) -> None:
        records = (
            (0, 2, 5, 4),
            (30, 1, 0, 0),
            (2, 0, 3, 4),
        )
        f1 = Frame.from_records(
            records, columns=('a', 'b', 'c', 'd'), index=('x', 'y', 'z')
        )

        self.assertEqual(
            f1.iloc_notfalsy_first(axis=1).to_pairs(), (('x', 1), ('y', 0), ('z', 0))
        )
        self.assertEqual(
            f1.iloc_notfalsy_last(axis=1).to_pairs(), (('x', 3), ('y', 1), ('z', 3))
        )

    # ---------------------------------------------------------------------------

    def test_frame_cov_a(self) -> None:
        f1 = Frame.from_dict(
            dict(a=(3, 2, 1), b=(4, 5, 6)), index=('x', 'y', 'z'), name='f2'
        )

        f2 = f1.cov()
        self.assertEqual(
            f2.to_pairs(),
            (('a', (('a', 1.0), ('b', -1.0))), ('b', (('a', -1.0), ('b', 1.0)))),
        )

        f3 = f1.cov(axis=0)
        self.assertEqual(
            f3.to_pairs(),
            (
                ('x', (('x', 0.5), ('y', 1.5), ('z', 2.5))),
                ('y', (('x', 1.5), ('y', 4.5), ('z', 7.5))),
                ('z', (('x', 2.5), ('y', 7.5), ('z', 12.5))),
            ),
        )

    def test_frame_cov_b(self) -> None:
        f1 = FrameGO.from_dict(
            dict(a=(3, 2, 1), b=(4, 5, 6)), index=('x', 'y', 'z'), name='f1'
        )

        f2 = f1.cov()
        self.assertEqual(
            f2.to_pairs(),
            (('a', (('a', 1.0), ('b', -1.0))), ('b', (('a', -1.0), ('b', 1.0)))),
        )

        f3 = f1.cov(axis=0)
        self.assertEqual(
            f3.to_pairs(),
            (
                ('x', (('x', 0.5), ('y', 1.5), ('z', 2.5))),
                ('y', (('x', 1.5), ('y', 4.5), ('z', 7.5))),
                ('z', (('x', 2.5), ('y', 7.5), ('z', 12.5))),
            ),
        )

        self.assertEqual(f3.name, 'f1')

    # ---------------------------------------------------------------------------
    def test_frame_corr_a(self) -> None:
        # >>> pd.DataFrame([[1, 2, 10, 0], [8, 4, 2, 0], [20, 24, 16, 1]], index=('a', 'b', 'c'), columns=('x', 'y', 'z', 'w')).corr()
        #           x         y         z         w
        # x  1.000000  0.958094  0.558094  0.931305
        # y  0.958094  1.000000  0.772401  0.996616
        # z  0.558094  0.772401  1.000000  0.821995
        # w  0.931305  0.996616  0.821995  1.000000
        f1 = sf.Frame.from_records(
            [[1, 2, 10, 0], [8, 4, 2, 0], [20, 24, 16, 1]],
            index=('a', 'b', 'c'),
            columns=('x', 'y', 'z', 'w'),
        ).corr()
        self.assertEqual(
            round(f1, 6).to_pairs(),
            (
                ('x', (('x', 1.0), ('y', 0.958094), ('z', 0.558094), ('w', 0.931305))),
                ('y', (('x', 0.958094), ('y', 1.0), ('z', 0.772401), ('w', 0.996616))),
                ('z', (('x', 0.558094), ('y', 0.772401), ('z', 1.0), ('w', 0.821995))),
                ('w', (('x', 0.931305), ('y', 0.996616), ('z', 0.821995), ('w', 1.0))),
            ),
        )

    def test_frame_corr_b(self) -> None:
        f1 = sf.Frame.from_records(
            [[1, 2, 10, 0], [8, 4, 2, 0], [20, 24, 16, 1]],
            index=('a', 'b', 'c'),
            columns=('x', 'y', 'z', 'w'),
        ).corr(axis=0)
        self.assertEqual(
            round(f1, 6).to_pairs(),
            (
                ('a', (('a', 1.0), ('b', -0.202714), ('c', 0.215843))),
                ('b', (('a', -0.202714), ('b', 1.0), ('c', 0.723736))),
                ('c', (('a', 0.215843), ('b', 0.723736), ('c', 1.0))),
            ),
        )

    # ---------------------------------------------------------------------------

    def test_frame_bloc_a1(self) -> None:
        f1 = Frame.from_dict(
            dict(a=(3, 2, 1), b=(4, 5, 6)), index=('x', 'y', 'z'), name='f2'
        )
        f3 = Frame.from_records(
            ((10, 20, 50, 60), (50.0, 60.4, -50, -60)),
            index=('p', 'q'),
            columns=IndexHierarchy.from_product(('I', 'II'), ('a', 'b')),
            name='f3',
        )

        s1 = f1.bloc[(f1 <= 2) | (f1 > 4)]
        self.assertEqual(
            s1.to_pairs(),
            ((('y', 'a'), 2), (('y', 'b'), 5), (('z', 'a'), 1), (('z', 'b'), 6)),
        )

        s3 = f3.bloc[f3 < 11]
        self.assertEqual(
            s3.to_pairs(),
            (
                (('p', 'I', 'a'), 10.0),
                (('q', 'II', 'a'), -50.0),
                (('q', 'II', 'b'), -60.0),
            ),
        )

    def test_frame_bloc_a2(self) -> None:
        f2 = sf.Frame.from_dict(
            dict(x=(1, 2, -5, 200), y=(3, 4, -5, -3000)),
            index=sf.IndexHierarchy.from_product(('I', 'II'), ('a', 'b')),
            name='f1',
        )
        s2 = f2.bloc[(f2 < 0)]
        self.assertEqual(
            s2.to_pairs(),
            ((('II', 'a', 'x'), -5), (('II', 'a', 'y'), -5), (('II', 'b', 'y'), -3000)),
        )

    def test_frame_bloc_b(self) -> None:
        f = sf.Frame.from_records(
            [[True, False], [False, True]], index=('a', 'b'), columns=['d', 'c']
        )
        self.assertEqual(
            f.assign.bloc[f]('T').assign.bloc[~f]('').to_pairs(),
            (('d', (('a', 'T'), ('b', ''))), ('c', (('a', ''), ('b', 'T')))),
        )

    def test_frame_bloc_c(self) -> None:
        f = sf.Frame.from_records(
            [[False, False, False], [False, False, False]],
            index=('a', 'b'),
            columns=['x', 'y', 'z'],
        )

        s1 = f.bloc[f == True]
        self.assertEqual(len(s1), 0)
        self.assertEqual(
            s1.index.dtypes.values.tolist(), [np.dtype('float64'), np.dtype('float64')]
        )

        s2 = f.bloc[f == False]
        self.assertEqual(
            s2.to_pairs(),
            (
                (('a', 'x'), False),
                (('a', 'y'), False),
                (('a', 'z'), False),
                (('b', 'x'), False),
                (('b', 'y'), False),
                (('b', 'z'), False),
            ),
        )

    def test_frame_bloc_d(self) -> None:
        f1 = Frame(np.arange(9).reshape(3, 3))
        f2 = Frame(np.arange(9).reshape(3, 3))
        # this forces a different block structure that resulted in different bloc ordering
        f2 = f2.assign[:].apply_element(lambda v: {v})
        sel = f1 % 1 == 0
        s1 = f1.bloc[sel]
        s2 = f2.bloc[sel]
        self.assertTrue(s1.index.equals(s2.index))

    # ---------------------------------------------------------------------------

    def test_frame_unset_index_a1(self) -> None:
        records = (
            (2, 2),
            (30, 3),
            (2, -95),
        )
        f1 = Frame.from_records(records, columns=('a', 'b'), index=('x', 'y', 'z'))
        self.assertEqual(
            f1.unset_index().to_pairs(),
            (
                ('__index0__', ((0, 'x'), (1, 'y'), (2, 'z'))),
                ('a', ((0, 2), (1, 30), (2, 2))),
                ('b', ((0, 2.0), (1, 3), (2, -95.0))),
            ),
        )

    def test_frame_unset_index_a2(self) -> None:
        records = (
            (2, 2),
            (30, 3),
        )
        f1 = Frame.from_records(
            records,
            columns=('a', 'b'),
            index=IndexHierarchy.from_labels(
                (
                    (1, 10),
                    (1, 20),
                )
            ),
        )
        with self.assertRaises(RuntimeError):
            _ = f1.unset_index(names=('a',))

    def test_frame_unset_index_b(self) -> None:
        records = (
            (1, 2, 'a', False),
            (30, 34, 'b', True),
            (54, 95, 'c', False),
            (65, 73, 'd', True),
        )
        columns = IndexHierarchy.from_product(('a', 'b'), (1, 2))
        index = IndexHierarchy.from_product((100, 200), (True, False))
        f1 = Frame.from_records(records, columns=columns, index=index)

        with self.assertRaises(RuntimeError):
            f1.unset_index()

    def test_frame_unset_index_c(self) -> None:
        records = (
            (1, 2, 'a', False),
            (30, 34, 'b', True),
            (54, 95, 'c', False),
            (65, 73, 'd', True),
        )
        index = IndexHierarchy.from_product((100, 200), (True, False), name=('a', 'b'))
        f1 = Frame.from_records(records, index=index)
        self.assertEqual(
            f1.unset_index().to_pairs(),
            (
                ('a', ((0, 100), (1, 100), (2, 200), (3, 200))),
                ('b', ((0, True), (1, False), (2, True), (3, False))),
                (0, ((0, 1), (1, 30), (2, 54), (3, 65))),
                (1, ((0, 2), (1, 34), (2, 95), (3, 73))),
                (2, ((0, 'a'), (1, 'b'), (2, 'c'), (3, 'd'))),
                (3, ((0, False), (1, True), (2, False), (3, True))),
            ),
        )

    def test_frame_unset_index_d(self) -> None:
        records = (
            (2, 2),
            (30, 3),
            (2, -95),
        )
        f1 = Frame.from_records(records, columns=('a', 'b'), index=('x', 'y', 'z'))
        f2 = f1.unset_index(names=('index',))

        self.assertEqual(
            f2.to_pairs(),
            (
                ('index', ((0, 'x'), (1, 'y'), (2, 'z'))),
                ('a', ((0, 2), (1, 30), (2, 2))),
                ('b', ((0, 2), (1, 3), (2, -95))),
            ),
        )

    def test_frame_unset_index_e(self) -> None:
        # using ILoc after unset led to an error because of no handling when loc is iloc
        f1 = sf.Frame.from_records(['a', 'b'], index=tuple(('c', 'd')))
        self.assertEqual(f1.loc['d', 0], 'b')
        self.assertEqual(f1.loc[sf.ILoc[0], 0], 'a')

        f2 = f1.unset_index()
        self.assertEqual(
            f2.to_pairs(),
            (('__index0__', ((0, 'c'), (1, 'd'))), (0, ((0, 'a'), (1, 'b')))),
        )

        self.assertEqual(f2.loc[0, 0], 'a')
        self.assertEqual(f2.loc[sf.ILoc[0], 0], 'a')

    @skip_win
    def test_frame_unset_index_f(self) -> None:
        records = (
            (2, 2),
            (30, 3),
            (2, -95),
        )
        f1 = Frame.from_records(
            records,
            columns=('a', 'b'),
        )
        f2 = f1.unset_index(names=('index',), consolidate_blocks=True)
        self.assertEqual(f2._blocks.shapes.tolist(), [(3, 3)])

    def test_frame_unset_index_g(self) -> None:
        records = (
            (12, True, 'x'),
            (13, False, 'y'),
            (-23, True, 'z'),
        )
        f1 = sf.Frame.from_records(records, columns=('A', 'B', 'C'), index=(5, 6, 7))
        f2 = f1.unset_index(names=('index',))
        self.assertEqual(
            f2.to_pairs(),
            (
                ('index', ((0, 5), (1, 6), (2, 7))),
                ('A', ((0, 12.0), (1, 13.0), (2, -23))),
                ('B', ((0, True), (1, False), (2, True))),
                ('C', ((0, 'x'), (1, 'y'), (2, 'z'))),
            ),
        )

    def test_frame_unset_index_h(self) -> None:
        records = (
            (1, 'w', True, 20.15),
            (2, 'x', False, -13.55),
            (3, 'y', False, 1.12),
            (4, 'z', False, 0.01),
        )
        index = IndexHierarchy.from_product(('a', 'b'), (0, 1))
        f1 = sf.Frame.from_records(records, index=index)

        f2 = f1.unset_index(names=('idx_1', 'idx_2'))
        self.assertEqual(
            f2.to_pairs(),
            (
                ('idx_1', ((0, 'a'), (1, 'a'), (2, 'b'), (3, 'b'))),
                ('idx_2', ((0, 0), (1, 1), (2, 0), (3, 1))),
                (0, ((0, 1), (1, 2), (2, 3), (3, 4))),
                (1, ((0, 'w'), (1, 'x'), (2, 'y'), (3, 'z'))),
                (2, ((0, True), (1, False), (2, False), (3, False))),
                (3, ((0, 20.15), (1, -13.55), (2, 1.12), (3, 0.01))),
            ),
        )

    def test_frame_unset_index_i(self) -> None:
        records = ((2, 2), (30, 3), (2, -95))
        f1 = Frame.from_records(
            records,
            columns=('a', 'b'),
        )
        f2 = f1.unset_index(drop=True)
        self.assertEqual(f2.shape, (3, 2))

    def test_frame_unset_index_j(self) -> None:
        records = ((2, 2), (30, 3), (2, -95))
        f1 = sf.Frame.from_records(records, columns=('a', 'b'))
        with self.assertRaises(RuntimeError):
            f1.unset_index(names=('unset_idx',), drop=True)

    def test_frame_unset_index_k(self) -> None:
        records = ((2, 2), (30, 3), (2, -95))
        f1 = Frame.from_records(
            records, columns=('a', 'b'), index=('idx_1', 'idx_2', 'idX_3')
        )
        f2 = f1.unset_index(drop=True)
        self.assertEqual(
            f2.to_pairs(),
            (('a', ((0, 2), (1, 30), (2, 2))), ('b', ((0, 2), (1, 3), (2, -95)))),
        )

    def test_frame_unset_index_l(self) -> None:
        records = ((7, 'a'), (7, 'b'), (7, 'c'))
        f_go = FrameGO.from_records(
            records, columns=('col1', 'col2'), index=('x', 'y', 'z')
        )
        f_immutable = f_go.unset_index(drop=True)
        f_go['col3'] = [True, False, False]
        # Ensure Frame does not get mutated after call and growing FrameGO
        self.assertEqual(
            f_immutable.to_pairs(),
            (
                ('col1', ((0, 7), (1, 7), (2, 7))),
                ('col2', ((0, 'a'), (1, 'b'), (2, 'c'))),
            ),
        )

    def test_unset_index_column_hierarchy(self) -> None:
        f = ff.parse('s(5,5)|i(I,str)|c(IH,(str,str))').rename(
            index='index_name', columns=('l1', 'l2')
        )
        unset = f.unset_index(names=[('outer', f.index.name)])
        assert unset.columns.values.tolist() == [
            ['outer', 'index_name'],
            ['zZbu', 'zOyq'],
            ['zZbu', 'zIA5'],
            ['ztsv', 'zGDJ'],
            ['ztsv', 'zmhG'],
            ['zUvW', 'zo2Q'],
        ]

        # A frame with hierarchical index and columns.
        f = ff.parse('s(5,5)|i(IH,(str,str))|c(IH,(str,str))').rename(
            index=('index_name1', 'index_name2'), columns=('l1', 'l2')
        )
        unset = f.unset_index(names=[('outer', n) for n in f.index.names])
        assert unset.columns.values.tolist() == [
            ['outer', 'index_name1'],
            ['outer', 'index_name2'],
            ['zZbu', 'zOyq'],
            ['zZbu', 'zIA5'],
            ['ztsv', 'zGDJ'],
            ['ztsv', 'zmhG'],
            ['zUvW', 'zo2Q'],
        ]

    def test_unset_index_column_hierarchy_w_dates(self) -> None:
        f = ff.parse('s(3,3)|i(I,str)|c((I, IY, I),(str,dtY,tdD))').rename(
            index='index_name',
            columns=('l1', 'l2'),
        )
        unset = f.unset_index(names=[('outer', 'middle', f.index.name)])
        assert unset.columns.values.tolist() == [
            ['outer', 'middle', 'index_name'],
            ['zZbu', 105269, datetime.timedelta(days=58768)],
            ['zZbu', 105269, datetime.timedelta(days=146284)],
            ['zZbu', 119909, datetime.timedelta(days=170440)],
        ]
        assert unset.columns.dtypes.values.tolist() == [
            np.dtype('<U5'),
            np.dtype('O'),
            np.dtype('O'),
        ]

        # dtypes should be preserved when possible.
        dt = f.columns.values_at_depth(1)[1]
        td = f.columns.values_at_depth(2)[1]
        unset2 = f.unset_index(
            names=[(f.index.name, dt, td)],
            columns_constructors=(Index, IndexYear, Index),
        )

        assert unset2.columns.values.tolist() == [
            ['index_name', np.datetime64('107239'), np.timedelta64(146284, 'D')],
            ['zZbu', np.datetime64('107239'), np.timedelta64(58768, 'D')],
            ['zZbu', np.datetime64('107239'), np.timedelta64(146284, 'D')],
            ['zZbu', np.datetime64('121879'), np.timedelta64(170440, 'D')],
        ]

        assert unset2.columns.dtypes.values.tolist() == [
            np.dtype('<U10'),
            np.dtype('<M8[Y]'),
            np.dtype('<m8[D]'),
        ]

    # ---------------------------------------------------------------------------
    @skip_win
    def test_frame_pivot_a(self) -> None:
        index = IndexHierarchy.from_product(
            ('far', 'near'), ('up', 'down'), ('left', 'right'), name=('z', 'y', 'x')
        )
        f1 = FrameGO(index=index)
        f1['a'] = range(len(f1))
        f1['b'] = (len(str(f1.index.values[i])) for i in range(len(f1)))

        f2 = f1.unset_index()
        post = f2.pivot(
            index_fields=('x', 'y'),  # values in this field become the index
            columns_fields=('z',),  # values in this field become columns
            data_fields=('a', 'b'),
        )
        self.assertEqual(post.columns.name, ('z', 'values'))
        self.assertEqual(post.index.name, ('x', 'y'))
        self.assertEqual(
            post.dtypes.values.tolist(),
            [
                np.dtype('int64'),
                np.dtype('int64'),
                np.dtype('int64'),
                np.dtype('int64'),
            ],
        )

        self.assertEqual(
            post.to_pairs(),
            (
                (
                    ('far', 'a'),
                    (
                        (('left', 'down'), 2),
                        (('left', 'up'), 0),
                        (('right', 'down'), 3),
                        (('right', 'up'), 1),
                    ),
                ),
                (
                    ('far', 'b'),
                    (
                        (('left', 'down'), 21),
                        (('left', 'up'), 19),
                        (('right', 'down'), 22),
                        (('right', 'up'), 20),
                    ),
                ),
                (
                    ('near', 'a'),
                    (
                        (('left', 'down'), 6),
                        (('left', 'up'), 4),
                        (('right', 'down'), 7),
                        (('right', 'up'), 5),
                    ),
                ),
                (
                    ('near', 'b'),
                    (
                        (('left', 'down'), 22),
                        (('left', 'up'), 20),
                        (('right', 'down'), 23),
                        (('right', 'up'), 21),
                    ),
                ),
            ),
        )

    def test_frame_pivot_b(self) -> None:
        index = IndexHierarchy.from_product(
            ('far', 'near'), ('up', 'down'), ('left', 'right'), name=('z', 'y', 'x')
        )
        f1 = FrameGO(index=index)
        f1['a'] = range(len(f1))
        f1['b'] = (len(str(f1.index.values[i])) for i in range(len(f1)))

        f2 = f1.unset_index()
        post = f2.pivot(
            index_fields=('x', 'y'),  # values in this field become the index
            columns_fields=('z',),  # values in this field become columns
            data_fields=('a',),
        )
        self.assertEqual(post.index.name, ('x', 'y'))
        self.assertEqual(post.columns.name, 'z')
        self.assertEqual(
            post.to_pairs(),
            (
                (
                    'far',
                    (
                        (('left', 'down'), 2),
                        (('left', 'up'), 0),
                        (('right', 'down'), 3),
                        (('right', 'up'), 1),
                    ),
                ),
                (
                    'near',
                    (
                        (('left', 'down'), 6),
                        (('left', 'up'), 4),
                        (('right', 'down'), 7),
                        (('right', 'up'), 5),
                    ),
                ),
            ),
        )

    def test_frame_pivot_c(self) -> None:
        index = IndexHierarchy.from_product(
            ('far', 'near'), ('up', 'down'), ('left', 'right'), name=('z', 'y', 'x')
        )
        f1 = FrameGO(index=index)
        f1['a'] = range(len(f1))
        f1['b'] = (len(str(f1.index.values[i])) for i in range(len(f1)))

        f2 = f1.unset_index()
        post = f2.pivot(
            index_fields=('x', 'y'),  # values in this field become the index
            data_fields=('b',),
        )

        self.assertEqual(post.index.name, ('x', 'y'))
        self.assertEqual(
            post.to_pairs(),
            (
                (
                    'b',
                    (
                        (('left', 'down'), 43),
                        (('left', 'up'), 39),
                        (('right', 'down'), 45),
                        (('right', 'up'), 41),
                    ),
                ),
            ),
        )

    def test_frame_pivot_d(self) -> None:
        index = IndexHierarchy.from_product(
            ('far', 'near'), ('up', 'down'), ('left', 'right'), name=('z', 'y', 'x')
        )
        f1 = FrameGO(index=index)
        f1['a'] = range(len(f1))
        f1['b'] = (len(str(f1.index.values[i])) for i in range(len(f1)))

        f2 = f1.unset_index()

        # single argument specs
        p1 = f2.pivot('z', 'x', 'a')
        self.assertEqual(p1.index.name, 'z')
        self.assertEqual(p1.columns.name, 'x')
        self.assertEqual(p1.__class__, FrameGO)
        self.assertEqual(
            p1.to_pairs(),
            (
                ('left', (('far', 2), ('near', 10))),
                ('right', (('far', 4), ('near', 12))),
            ),
        )

        p2 = f2.pivot('z', 'x', 'b')
        self.assertEqual(p2.index.name, 'z')
        self.assertEqual(p2.columns.name, 'x')
        self.assertEqual(
            p2.to_pairs(),
            (
                ('left', (('far', 40), ('near', 42))),
                ('right', (('far', 42), ('near', 44))),
            ),
        )

        p3 = f2.pivot('x', 'y', 'a')
        self.assertEqual(p3.index.name, 'x')
        self.assertEqual(p3.columns.name, 'y')
        self.assertEqual(
            p3.to_pairs(),
            (
                ('down', (('left', 8), ('right', 10))),
                ('up', (('left', 4), ('right', 6))),
            ),
        )

        p4 = f2.pivot('x', 'y', 'b')
        self.assertEqual(p4.index.name, 'x')
        self.assertEqual(p4.columns.name, 'y')
        self.assertEqual(
            p4.to_pairs(),
            (
                ('down', (('left', 43), ('right', 45))),
                ('up', (('left', 39), ('right', 41))),
            ),
        )

    @skip_win
    def test_frame_pivot_e1(self) -> None:
        index = IndexHierarchy.from_product(
            ('far', 'near'), ('up', 'down'), ('left', 'right'), name=('z', 'y', 'x')
        )
        f1 = FrameGO(index=index)
        f1['a'] = range(len(f1))
        f1['b'] = (len(str(f1.index.values[i])) for i in range(len(f1)))

        f2 = f1.unset_index()

        # no columns provided
        p1 = f2.pivot('z', data_fields='b')

        self.assertEqual(p1.index.name, 'z')
        self.assertEqual(p1.columns.name, 'values')
        self.assertEqual(p1.dtypes.values.tolist(), [np.dtype('int64')])
        self.assertEqual(p1.to_pairs(), (('b', (('far', 82), ('near', 86))),))

    def test_frame_pivot_e2(self) -> None:
        index = IndexHierarchy.from_product(
            ('far', 'near'), ('up', 'down'), ('left', 'right'), name=('z', 'y', 'x')
        )
        f1 = FrameGO(index=index)
        f1['a'] = range(len(f1))
        f1['b'] = (len(str(f1.index.values[i])) for i in range(len(f1)))

        f2 = f1.unset_index()
        self.assertEqual(
            f2.pivot('z', data_fields=('a', 'b')).to_pairs(),
            (('a', (('far', 6), ('near', 22))), ('b', (('far', 82), ('near', 86)))),
        )

    @skip_win
    def test_frame_pivot_f(self) -> None:
        index = IndexHierarchy.from_product(
            ('far', 'near'), ('up', 'down'), ('left', 'right'), name=('z', 'y', 'x')
        )
        f1 = FrameGO(index=index)
        f1['a'] = range(len(f1))
        f1['b'] = (len(str(f1.index.values[i])) for i in range(len(f1)))

        f2 = f1.unset_index()

        # shows unique values of 'b' as columns, then shows values for z, a
        post = f2.pivot(('x', 'y'), ('b',), fill_value='', func=None)

        self.assertEqual(post.index.name, ('x', 'y'))
        self.assertEqual(post.columns.name, ('b', 'values'))
        # NOTE: because we are filling na with empty strings, we get object dtypes
        self.assertEqual(
            post.dtypes.values.tolist(),
            [
                np.dtype('<U4'),
                np.dtype('O'),
                np.dtype('<U4'),
                np.dtype('O'),
                np.dtype('<U4'),
                np.dtype('O'),
                np.dtype('<U4'),
                np.dtype('O'),
                np.dtype('<U4'),
                np.dtype('O'),
            ],
        )
        self.assertEqual(
            post.to_pairs(),
            (
                (
                    (19, 'z'),
                    (
                        (('left', 'down'), ''),
                        (('left', 'up'), 'far'),
                        (('right', 'down'), ''),
                        (('right', 'up'), ''),
                    ),
                ),
                (
                    (19, 'a'),
                    (
                        (('left', 'down'), ''),
                        (('left', 'up'), 0),
                        (('right', 'down'), ''),
                        (('right', 'up'), ''),
                    ),
                ),
                (
                    (20, 'z'),
                    (
                        (('left', 'down'), ''),
                        (('left', 'up'), 'near'),
                        (('right', 'down'), ''),
                        (('right', 'up'), 'far'),
                    ),
                ),
                (
                    (20, 'a'),
                    (
                        (('left', 'down'), ''),
                        (('left', 'up'), 4),
                        (('right', 'down'), ''),
                        (('right', 'up'), 1),
                    ),
                ),
                (
                    (21, 'z'),
                    (
                        (('left', 'down'), 'far'),
                        (('left', 'up'), ''),
                        (('right', 'down'), ''),
                        (('right', 'up'), 'near'),
                    ),
                ),
                (
                    (21, 'a'),
                    (
                        (('left', 'down'), 2),
                        (('left', 'up'), ''),
                        (('right', 'down'), ''),
                        (('right', 'up'), 5),
                    ),
                ),
                (
                    (22, 'z'),
                    (
                        (('left', 'down'), 'near'),
                        (('left', 'up'), ''),
                        (('right', 'down'), 'far'),
                        (('right', 'up'), ''),
                    ),
                ),
                (
                    (22, 'a'),
                    (
                        (('left', 'down'), 6),
                        (('left', 'up'), ''),
                        (('right', 'down'), 3),
                        (('right', 'up'), ''),
                    ),
                ),
                (
                    (23, 'z'),
                    (
                        (('left', 'down'), ''),
                        (('left', 'up'), ''),
                        (('right', 'down'), 'near'),
                        (('right', 'up'), ''),
                    ),
                ),
                (
                    (23, 'a'),
                    (
                        (('left', 'down'), ''),
                        (('left', 'up'), ''),
                        (('right', 'down'), 7),
                        (('right', 'up'), ''),
                    ),
                ),
            ),
        )

    def test_frame_pivot_g(self) -> None:
        index = IndexHierarchy.from_product(
            ('far', 'near'), ('up', 'down'), ('left', 'right'), name=('z', 'y', 'x')
        )
        f1 = FrameGO(index=index)
        f1['a'] = range(len(f1))
        f1['b'] = (len(str(f1.index.values[i])) for i in range(len(f1)))

        f2 = f1.unset_index()

        # multiple columns; all reamining fields go to data_fields
        post1 = f2.pivot('z', ('y', 'x'))
        self.assertEqual(post1.index.name, 'z')
        self.assertEqual(post1.columns.name, ('y', 'x', 'values'))
        self.assertEqual(
            post1.to_pairs(),
            (
                (('down', 'left', 'a'), (('far', 2), ('near', 6))),
                (('down', 'left', 'b'), (('far', 21), ('near', 22))),
                (('down', 'right', 'a'), (('far', 3), ('near', 7))),
                (('down', 'right', 'b'), (('far', 22), ('near', 23))),
                (('up', 'left', 'a'), (('far', 0), ('near', 4))),
                (('up', 'left', 'b'), (('far', 19), ('near', 20))),
                (('up', 'right', 'a'), (('far', 1), ('near', 5))),
                (('up', 'right', 'b'), (('far', 20), ('near', 21))),
            ),
        )

    def test_frame_pivot_h(self) -> None:
        index = IndexHierarchy.from_product(
            ('far', 'near'), ('up', 'down'), ('left', 'right'), name=('z', 'y', 'x')
        )
        f1 = FrameGO(index=index)
        f1['a'] = range(len(f1))
        f1['b'] = (len(str(f1.index.values[i])) for i in range(len(f1)))

        f2 = f1.unset_index()

        # specifying a data_fields value
        post1 = f2.pivot('z', ('y', 'x'), 'a')
        self.assertEqual(post1.index.name, 'z')
        self.assertEqual(post1.columns.name, ('y', 'x'))
        self.assertEqual(
            post1.to_pairs(),
            (
                (('down', 'left'), (('far', 2), ('near', 6))),
                (('down', 'right'), (('far', 3), ('near', 7))),
                (('up', 'left'), (('far', 0), ('near', 4))),
                (('up', 'right'), (('far', 1), ('near', 5))),
            ),
        )

    def test_frame_pivot_i(self) -> None:
        index = IndexHierarchy.from_product(
            ('far', 'near'), ('up', 'down'), ('left', 'right'), name=('z', 'y', 'x')
        )
        f1 = FrameGO(index=index)
        f1['a'] = range(len(f1))
        f1['b'] = (len(str(f1.index.values[i])) for i in range(len(f1)))

        f2 = f1.unset_index()

        post1 = f2.pivot('z', 'y', 'a', func={'min': np.min, 'max': np.max})
        self.assertEqual(post1.index.name, 'z')
        self.assertEqual(post1.columns.name, ('y', 'func'))
        self.assertEqual(
            post1.to_pairs(),
            (
                (('down', 'min'), (('far', 2), ('near', 6))),
                (('down', 'max'), (('far', 3), ('near', 7))),
                (('up', 'min'), (('far', 0), ('near', 4))),
                (('up', 'max'), (('far', 1), ('near', 5))),
            ),
        )

    def test_frame_pivot_j1(self) -> None:
        index = IndexHierarchy.from_product(
            ('far', 'near'), ('up', 'down'), ('left', 'right'), name=('z', 'y', 'x')
        )
        f1 = FrameGO(index=index)
        f1['a'] = range(len(f1))
        f1['b'] = (len(str(f1.index.values[i])) for i in range(len(f1)))

        f2 = f1.unset_index()

        post1 = f2.pivot('z', ('y', 'x'), 'b', func={'min': np.min, 'max': np.max})
        self.assertEqual(post1.index.name, 'z')
        self.assertEqual(post1.columns.name, ('y', 'x', 'func'))

        self.assertEqual(
            post1.to_pairs(),
            (
                (('down', 'left', 'min'), (('far', 21), ('near', 22))),
                (('down', 'left', 'max'), (('far', 21), ('near', 22))),
                (('down', 'right', 'min'), (('far', 22), ('near', 23))),
                (('down', 'right', 'max'), (('far', 22), ('near', 23))),
                (('up', 'left', 'min'), (('far', 19), ('near', 20))),
                (('up', 'left', 'max'), (('far', 19), ('near', 20))),
                (('up', 'right', 'min'), (('far', 20), ('near', 21))),
                (('up', 'right', 'max'), (('far', 20), ('near', 21))),
            ),
        )

    def test_frame_pivot_j2(self) -> None:
        index = IndexHierarchy.from_product(
            ('far', 'near'), ('up', 'down'), ('left', 'right'), name=('z', 'y', 'x')
        )
        f1 = FrameGO(index=index)
        f1['a'] = range(len(f1))
        f1['b'] = (len(str(f1.index.values[i])) for i in range(len(f1)))

        f2 = f1.unset_index()

        # default populates data values for a, b
        post2 = f2.pivot('z', ('y', 'x'), func={'min': np.min, 'max': np.max})
        self.assertEqual(post2.index.name, 'z')
        self.assertEqual(post2.columns.name, ('y', 'x', 'values', 'func'))
        self.assertEqual(
            post2.to_pairs(),
            (
                (('down', 'left', 'a', 'min'), (('far', 2), ('near', 6))),
                (('down', 'left', 'a', 'max'), (('far', 2), ('near', 6))),
                (('down', 'left', 'b', 'min'), (('far', 21), ('near', 22))),
                (('down', 'left', 'b', 'max'), (('far', 21), ('near', 22))),
                (('down', 'right', 'a', 'min'), (('far', 3), ('near', 7))),
                (('down', 'right', 'a', 'max'), (('far', 3), ('near', 7))),
                (('down', 'right', 'b', 'min'), (('far', 22), ('near', 23))),
                (('down', 'right', 'b', 'max'), (('far', 22), ('near', 23))),
                (('up', 'left', 'a', 'min'), (('far', 0), ('near', 4))),
                (('up', 'left', 'a', 'max'), (('far', 0), ('near', 4))),
                (('up', 'left', 'b', 'min'), (('far', 19), ('near', 20))),
                (('up', 'left', 'b', 'max'), (('far', 19), ('near', 20))),
                (('up', 'right', 'a', 'min'), (('far', 1), ('near', 5))),
                (('up', 'right', 'a', 'max'), (('far', 1), ('near', 5))),
                (('up', 'right', 'b', 'min'), (('far', 20), ('near', 21))),
                (('up', 'right', 'b', 'max'), (('far', 20), ('near', 21))),
            ),
        )

    def test_frame_pivot_k(self) -> None:
        index = IndexHierarchy.from_product(
            ('far', 20), (None, 'down'), (False, 'right'), name=('z', 'y', 'x')
        )
        f1 = FrameGO(index=index)
        f1['a'] = range(len(f1))
        f1['b'] = (len(str(f1.index.values[i])) for i in range(len(f1)))

        f2 = f1.unset_index()
        post1 = f2.pivot('z', 'y', 'a')
        self.assertEqual(post1.index.name, 'z')
        self.assertEqual(post1.columns.name, 'y')
        self.assertEqual(
            post1.to_pairs(),
            ((None, (('far', 1), (20, 9))), ('down', (('far', 5), (20, 13)))),
        )

    def test_frame_pivot_m(self) -> None:
        index = IndexHierarchy.from_product(
            ('far', 20), (None, 'down'), (False, 'right'), name=('z', 'y', 'x')
        )
        f1 = FrameGO(index=index)
        f1['a'] = range(len(f1))

        f2 = f1.unset_index()
        post1 = f2.pivot('z', 'y', 'a', func=np.sum)
        self.assertEqual(
            post1.to_pairs(),
            ((None, (('far', 1), (20, 9))), ('down', (('far', 5), (20, 13)))),
        )

        with self.assertRaises(ErrorInitFrame):
            # no fields remain to populate data.
            _ = f2[['z', 'y']].pivot('z', 'y')  # type: ignore

        with self.assertRaises(ErrorInitFrame):
            # cannot create a pivot Frame from a field (q) that is not a column
            _ = f2.pivot('q')

    def test_frame_pivot_n1(self) -> None:
        f1 = FrameGO(index=range(3))
        f1['a'] = np.array(range(3)) + 10001
        f1['b'] = np.array(range(3), 'datetime64[D]')
        f1['c'] = np.array(range(3)) * 1e9

        f2 = f1.pivot('b', 'a', fill_value=0, index_constructor=IndexDate, func=np.nansum)
        self.assertIs(f2.index.__class__, sf.IndexDate)
        self.assertIs(f2.__class__, FrameGO)
        self.assertEqual(f2.index.name, 'b')
        self.assertEqual(
            f2.to_pairs(),
            (
                (
                    10001,
                    (
                        (np.datetime64('1970-01-01'), 0.0),
                        (np.datetime64('1970-01-02'), 0.0),
                        (np.datetime64('1970-01-03'), 0.0),
                    ),
                ),
                (
                    10002,
                    (
                        (np.datetime64('1970-01-01'), 0.0),
                        (np.datetime64('1970-01-02'), 1000000000.0),
                        (np.datetime64('1970-01-03'), 0.0),
                    ),
                ),
                (
                    10003,
                    (
                        (np.datetime64('1970-01-01'), 0.0),
                        (np.datetime64('1970-01-02'), 0.0),
                        (np.datetime64('1970-01-03'), 2000000000.0),
                    ),
                ),
            ),
        )

    def test_frame_pivot_n2(self) -> None:
        f1 = FrameGO(index=range(3))
        f1['a'] = np.array(range(3)) + 10001
        f1['b'] = np.array(range(3), 'datetime64[D]')
        f1['c'] = np.array(range(3)) * 1e9

        f2 = f1.pivot('b', 'a', fill_value='', index_constructor=IndexDate, func=str)
        dt64 = np.datetime64
        self.assertEqual(
            f2.to_pairs(),
            (
                (
                    10001,
                    (
                        (dt64('1970-01-01'), '[0.]'),
                        (dt64('1970-01-02'), ''),
                        (dt64('1970-01-03'), ''),
                    ),
                ),
                (
                    10002,
                    (
                        (dt64('1970-01-01'), ''),
                        (dt64('1970-01-02'), '[1.e+09]'),
                        (dt64('1970-01-03'), ''),
                    ),
                ),
                (
                    10003,
                    (
                        (dt64('1970-01-01'), ''),
                        (dt64('1970-01-02'), ''),
                        (dt64('1970-01-03'), '[2.e+09]'),
                    ),
                ),
            ),
        )

    def test_frame_pivot_n3(self) -> None:
        f1 = FrameGO(index=range(3))
        f1['a'] = np.array(range(3)) + 10001
        f1['b'] = np.array(range(3), 'datetime64[D]')
        f1['c'] = np.array(range(3)) * 1e9

        f2 = f1.pivot(
            index_fields='a',
            columns_fields='b',
            columns_constructor=IndexDateGO,
            fill_value=0,
        )
        self.assertEqual(f2.columns.__class__, IndexDateGO)
        self.assertEqual(
            f2.to_pairs(),
            (
                (
                    np.datetime64('1970-01-01'),
                    (
                        (np.int64(10001), np.float64(0.0)),
                        (np.int64(10002), np.float64(0.0)),
                        (np.int64(10003), np.float64(0.0)),
                    ),
                ),
                (
                    np.datetime64('1970-01-02'),
                    (
                        (np.int64(10001), np.float64(0.0)),
                        (np.int64(10002), np.float64(1000000000.0)),
                        (np.int64(10003), np.float64(0.0)),
                    ),
                ),
                (
                    np.datetime64('1970-01-03'),
                    (
                        (np.int64(10001), np.float64(0.0)),
                        (np.int64(10002), np.float64(0.0)),
                        (np.int64(10003), np.float64(2000000000.0)),
                    ),
                ),
            ),
        )

    def test_frame_pivot_o(self) -> None:
        index = IndexHierarchy.from_product(
            ('far', 'near'), ('up', 'down'), ('left', 'right'), name=('z', 'y', 'x')
        )
        f1 = FrameGO(index=index)
        f1['a'] = range(len(f1))
        f1['b'] = (len(str(f1.index.values[i])) for i in range(len(f1)))

        f2 = f1.unset_index()

        p1 = f2.pivot(
            'y',
            data_fields=('a', 'b'),
            func={'mean': np.mean, 'max': np.max, 'values': tuple},
        )

        self.assertEqual(
            p1.to_pairs(),
            (
                (('a', 'mean'), (('down', 4.5), ('up', 2.5))),
                (('a', 'max'), (('down', 7), ('up', 5))),
                (('a', 'values'), (('down', (2, 3, 6, 7)), ('up', (0, 1, 4, 5)))),
                (('b', 'mean'), (('down', 22.0), ('up', 20.0))),
                (('b', 'max'), (('down', 23), ('up', 21))),
                (
                    ('b', 'values'),
                    (('down', (21, 22, 22, 23)), ('up', (19, 20, 20, 21))),
                ),
            ),
        )

    def test_frame_pivot_p(self) -> None:
        index = IndexHierarchy.from_product(
            ('far', 'near'), ('up', 'down'), ('left', 'right'), name=('z', 'y', 'x')
        )
        f1 = FrameGO(index=index)
        f1['a'] = range(len(f1))
        f1['b'] = (len(str(f1.index.values[i])) for i in range(len(f1)))

        f2 = f1.unset_index()

        p1 = f2.pivot(
            'x', 'y', data_fields=('a', 'b'), func={'mean': np.mean, 'min': np.min}
        )

        self.assertEqual(
            p1.to_pairs(),
            (
                (('down', 'a', 'mean'), (('left', 4.0), ('right', 5.0))),
                (('down', 'a', 'min'), (('left', 2), ('right', 3))),
                (('down', 'b', 'mean'), (('left', 21.5), ('right', 22.5))),
                (('down', 'b', 'min'), (('left', 21), ('right', 22))),
                (('up', 'a', 'mean'), (('left', 2.0), ('right', 3.0))),
                (('up', 'a', 'min'), (('left', 0), ('right', 1))),
                (('up', 'b', 'mean'), (('left', 19.5), ('right', 20.5))),
                (('up', 'b', 'min'), (('left', 19), ('right', 20))),
            ),
        )

    def test_frame_pivot_q(self) -> None:
        f1 = sf.Frame.from_records([[0, 'A'], [1, None], [2, 'B']])
        f2 = f1.pivot(1)
        self.assertEqual(f2.to_pairs(), ((0, (('A', 0), (None, 1), ('B', 2))),))

    def test_frame_pivot_r(self) -> None:
        f1 = sf.Frame.from_records([[0, 'A', False], [1, None, True], [2, 'B', False]])
        f2 = f1.pivot((0, 1), func=np.all)

        self.assertEqual(f2.dtypes.values.tolist(), [np.dtype('bool')])

    def test_frame_pivot_s(self) -> None:
        f = sf.FrameGO(index=range(6))
        f['a'] = tuple('b' * 3 + 'a' * 3)
        f['b'] = tuple('b' * 2 + 'c' + 'a' * 3)
        f['c'] = range(6)

        f2 = f.pivot(('a', 'b'))
        self.assertEqual(
            f2.to_pairs(),
            (('c', ((('a', 'a'), 12), (('b', 'b'), 1), (('b', 'c'), 2))),),
        )

    def test_frame_pivot_t(self) -> None:
        index = IndexHierarchy.from_product(
            ('far', 'near'), ('up', 'down'), ('left', 'right'), name=('z', 'y', 'x')
        )
        f1 = FrameGO(index=index)
        f1['a'] = range(len(f1))
        f1['b'] = (len(str(f1.index.values[i])) for i in range(len(f1)))
        f2 = f1.unset_index()

        post1 = f2.pivot(
            index_fields=('z', 'x', 'y'),  # values in this field become the index
            data_fields=('a', 'b'),
            func=None,
        )
        self.assertEqual(
            post1.to_pairs(),
            (
                (
                    'a',
                    (
                        (('far', 'left', 'down'), 2),
                        (('far', 'left', 'up'), 0),
                        (('far', 'right', 'down'), 3),
                        (('far', 'right', 'up'), 1),
                        (('near', 'left', 'down'), 6),
                        (('near', 'left', 'up'), 4),
                        (('near', 'right', 'down'), 7),
                        (('near', 'right', 'up'), 5),
                    ),
                ),
                (
                    'b',
                    (
                        (('far', 'left', 'down'), 21),
                        (('far', 'left', 'up'), 19),
                        (('far', 'right', 'down'), 22),
                        (('far', 'right', 'up'), 20),
                        (('near', 'left', 'down'), 22),
                        (('near', 'left', 'up'), 20),
                        (('near', 'right', 'down'), 23),
                        (('near', 'right', 'up'), 21),
                    ),
                ),
            ),
        )
        post2 = f2.pivot(
            index_fields=('z', 'x', 'y'),  # values in this field become the index
            data_fields=('a', 'b'),
            func=np.nansum,
        )
        self.assertTrue(post1.equals(post2))

    def test_frame_pivot_u1(self) -> None:
        index = IndexHierarchy.from_product(
            ('far', 'near'), ('up', 'down'), ('left', 'right'), name=('z', 'y', 'x')
        )
        f1 = FrameGO(index=index)
        f1['a'] = range(len(f1))
        f1['b'] = (len(str(f1.index.values[i])) for i in range(len(f1)))
        f2 = f1.unset_index()

        with self.assertRaises(RuntimeError):
            post1 = f2.pivot(
                index_fields=('z', 'x'),  # values in this field become the index
                data_fields=('a', 'b'),
                func=None,
            )

    def test_frame_pivot_u2(self) -> None:
        index = IndexHierarchy.from_product(
            ('far', 'near'), ('up', 'down'), ('left', 'right'), name=('z', 'y', 'x')
        )
        f1 = FrameGO(index=index)
        f1['a'] = range(len(f1))
        f1['b'] = (len(str(f1.index.values[i])) for i in range(len(f1)))
        f2 = f1.unset_index()
        with self.assertRaises(RuntimeError):
            post1 = f2.pivot(
                index_fields='z',
                columns_fields='y',
                data_fields=('a', 'b'),
                func=None,
            )

    def test_frame_pivot_v(self) -> None:
        f1 = FrameGO(index=IndexAutoFactory(4))
        f1['a'] = np.arange(4)
        f1['b'] = np.arange(4) * 10
        f1['c'] = np.arange(4) * 100

        post = f1.pivot(index_fields='a', columns_fields='b', fill_value=0, func=None)
        self.assertEqual(
            post.to_pairs(),
            (
                (0, ((0, 0), (1, 0), (2, 0), (3, 0))),
                (10, ((0, 0), (1, 100), (2, 0), (3, 0))),
                (20, ((0, 0), (1, 0), (2, 200), (3, 0))),
                (30, ((0, 0), (1, 0), (2, 0), (3, 300))),
            ),
        )

    def test_frame_pivot_w(self) -> None:
        f1 = sf.Frame.from_records([[0, 'A'], [1, None], [2, 'B']])

        # NOTE: order is different with func=None as we avoid a group-by sort
        f2 = f1.pivot(1, func=None)
        self.assertEqual(f2.to_pairs(), ((0, (('A', 0), (None, 1), ('B', 2))),))

        f3 = sf.Frame.from_records([[0, 'A', 10], [1, None, 20], [2, 'B', 30]])
        f4 = f3.pivot(1, func=None)
        self.assertEqual(
            f4.to_pairs(),
            (
                (0, (('A', 0), (None, 1), ('B', 2))),
                (2, (('A', 10), (None, 20), ('B', 30))),
            ),
        )

    @skip_win
    def test_frame_pivot_x(self) -> None:
        f = (
            ff.parse('s(10,4)|v(int)')
            .assign[0]
            .apply(lambda x: x % 3)
            .assign[1]
            .apply(lambda x: x % 3)
            .assign[2]
            .apply(lambda x: x % 3)
        )
        # remove the one duplicated value so no aggregation is needed
        f = f.drop.loc[[3, 5, 9]]
        f2 = f.pivot(index_fields=(0, 1), columns_fields=2, func=None, fill_value=0)
        f3 = f.pivot(index_fields=(0, 1), columns_fields=2, fill_value=0)
        self.assertTrue(
            f2.equals(f3, compare_name=True, compare_dtype=True, compare_class=True)
        )
        self.assertEqual(
            f2.to_pairs(),
            (
                (
                    0,
                    (
                        ((0, 2), 129017),
                        ((1, 0), 0),
                        ((1, 1), 0),
                        ((1, 2), 0),
                        ((2, 0), 0),
                    ),
                ),
                (
                    1,
                    (
                        ((0, 2), 0),
                        ((1, 0), 119909),
                        ((1, 1), 0),
                        ((1, 2), 166924),
                        ((2, 0), 0),
                    ),
                ),
                (
                    2,
                    (
                        ((0, 2), 0),
                        ((1, 0), 194224),
                        ((1, 1), 172133),
                        ((1, 2), 197228),
                        ((2, 0), 35021),
                    ),
                ),
            ),
        )

    def test_frame_pivot_y1(self) -> None:
        f1 = (
            ff.parse('s(10,4)|v(int64)')
            .assign[0]
            .apply(lambda x: x % 3)
            .assign[1]
            .apply(lambda x: x % 3)
            .assign[2]
            .apply(lambda x: x % 3)
        )

        f2 = f1.pivot(index_fields=2, columns_fields=1, data_fields=[0, 3])
        self.assertEqual(
            [(k, v.kind) for k, v in f2.dtypes.items()],
            [
                ((0, 0), 'f'),
                ((0, 3), 'f'),
                ((1, 0), 'f'),
                ((1, 3), 'f'),
                ((2, 0), 'i'),
                ((2, 3), 'i'),
            ],
        )

        f3 = f1.pivot(
            index_fields=2, columns_fields=1, data_fields=[0, 3], func=len, fill_value=0
        )
        self.assertEqual(f3.index.name, 2)
        self.assertEqual(f3.columns.name, (1, 'values'))
        self.assertEqual(
            f3.to_pairs(),
            (
                ((0, 0), ((0, 0), (1, 1), (2, 4))),
                ((0, 3), ((0, 0), (1, 1), (2, 4))),
                ((1, 0), ((0, 0), (1, 0), (2, 1))),
                ((1, 3), ((0, 0), (1, 0), (2, 1))),
                ((2, 0), ((0, 1), (1, 1), (2, 2))),
                ((2, 3), ((0, 1), (1, 1), (2, 2))),
            ),
        )

    # NOTE: this is failing on old Python, windows with Windows fatal exception: access violation
    @skip_win
    def test_frame_pivot_y2(self) -> None:
        f1 = (
            ff.parse('s(10,4)|v(int)')
            .assign[0]
            .apply(lambda x: x % 3)
            .assign[1]
            .apply(lambda x: x % 3)
            .assign[2]
            .apply(lambda x: x % 3)
        )

        f2 = f1.pivot(
            index_fields=2,
            columns_fields=1,
            data_fields=0,
            func=str,
            fill_value='[]',
        )
        self.assertEqual(
            f2.to_pairs(),
            (
                (0, ((0, '[]'), (1, '[1]'), (2, '[2 2 1 2]'))),
                (1, ((0, '[]'), (1, '[]'), (2, '[1]'))),
                (2, ((0, '[0]'), (1, '[1]'), (2, '[1 1]'))),
            ),
        )

    def test_frame_pivot_y3(self) -> None:
        f1 = (
            ff.parse('s(10,4)|v(int)')
            .assign[0]
            .apply(lambda x: x % 3)
            .assign[1]
            .apply(lambda x: x % 3)
            .assign[2]
            .apply(lambda x: x % 3)
        )
        with self.assertRaises(InvalidFillValue):
            f1.pivot(index_fields=2, fill_value=FillValueAuto)

    def test_frame_pivot_z1(self) -> None:
        f1 = (
            ff.parse('s(1_000,3)|v(int)')
            .assign[0]
            .apply(lambda s: s % 4)
            .assign[1]
            .apply(lambda s: s % 7)
            .relabel(columns=('a', 'b', 'c'))
        )
        f2 = f1.pivot(index_fields='a', columns_fields='b')
        self.assertEqual(f2.shape, (4, 7))
        self.assertEqual(
            f2.to_pairs(),
            (
                (0, ((0, 475193.0), (1, 570172.0), (2, 1796105.0), (3, 1492263.0))),
                (1, ((0, 360431.0), (1, 1405373.0), (2, 1236720.0), (3, 1400981.0))),
                (2, ((0, 2148747.0), (1, 1342595.0), (2, 931718.0), (3, 1923485.0))),
                (3, ((0, 2231329.0), (1, 465616.0), (2, 1372863.0), (3, 1699582.0))),
                (4, ((0, 2691795.0), (1, 1356194.0), (2, 676459.0), (3, 469668.0))),
                (5, ((0, 1193332.0), (1, 1318408.0), (2, 141940.0), (3, 1193659.0))),
                (6, ((0, 659383.0), (1, 1278871.0), (2, 2429167.0), (3, -408548.0))),
            ),
        )
        # import ipdb; ipdb.set_trace()

    # ---------------------------------------------------------------------------

    def test_frame_bool_a(self) -> None:
        records = (
            (2, 2),
            (30, 3),
            (2, -95),
        )
        f1 = Frame.from_records(records, columns=('a', 'b'), index=('x', 'y', 'z'))
        with self.assertRaises(ValueError):
            bool(f1)

    def test_frame_bool_b(self) -> None:
        f1 = Frame(columns=('a', 'b'))

        with self.assertRaises(ValueError):
            bool(f1)

    # ---------------------------------------------------------------------------

    def test_frame_frame_assign_a(self) -> None:
        f1 = Frame(columns=('a', 'b'))

        fa1 = FrameAssignILoc(f1, key=(0, 0))
        fa2 = FrameAssignBLoc(f1, key=f1)

    # ---------------------------------------------------------------------------

    def test_frame_any_a(self) -> None:
        records = (
            (2, 2),
            (30, 0),
            (2, -95),
        )
        f1 = Frame.from_records(records, columns=('a', 'b'), index=('x', 'y', 'z'))

        self.assertEqual(f1.all().to_pairs(), (('a', True), ('b', False)))
        self.assertEqual(f1.any().to_pairs(), (('a', True), ('b', True)))

        self.assertEqual(
            f1.all(axis=1).to_pairs(), (('x', True), ('y', False), ('z', True))
        )
        self.assertEqual(
            f1.any(axis=1).to_pairs(), (('x', True), ('y', True), ('z', True))
        )

    def test_frame_any_b(self) -> None:
        records = (
            (2, 2),
            (np.nan, 0),
            (np.nan, -95),
        )
        f1 = Frame.from_records(records, columns=('a', 'b'), index=('x', 'y', 'z'))

        self.assertEqual(f1.all().to_pairs(), (('a', True), ('b', False)))
        self.assertEqual(f1.any().to_pairs(), (('a', True), ('b', True)))

        self.assertEqual(
            f1.all(axis=1).to_pairs(), (('x', True), ('y', False), ('z', True))
        )
        self.assertEqual(
            f1.any(axis=1).to_pairs(), (('x', True), ('y', False), ('z', True))
        )

    def test_frame_any_c(self) -> None:
        records = (
            (2, 2),
            (np.nan, 0),
            (np.nan, -95),
        )
        f1 = Frame.from_records(records, columns=('a', 'b'), index=('x', 'y', 'z'))

        self.assertEqual(f1.all(skipna=False).to_pairs(), (('a', True), ('b', False)))

        self.assertEqual(f1.any(skipna=False).to_pairs(), (('a', True), ('b', True)))

        self.assertEqual(
            f1.all(axis=1, skipna=False).to_pairs(),
            (('x', True), ('y', False), ('z', True)),
        )

        self.assertEqual(
            f1.any(axis=1, skipna=False).to_pairs(),
            (('x', True), ('y', True), ('z', True)),
        )

    # ---------------------------------------------------------------------------

    def test_frame_round_a(self) -> None:
        a1 = np.full(
            4,
            0.33333,
        )
        a2 = np.full(
            (4, 2),
            0.88888,
        )
        a3 = np.full(4, 0.55555)
        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        f1 = Frame(tb1)
        f2 = round(f1)

        self.assertEqual(
            f2.to_pairs(),
            (
                (0, ((0, 0.0), (1, 0.0), (2, 0.0), (3, 0.0))),
                (1, ((0, 1.0), (1, 1.0), (2, 1.0), (3, 1.0))),
                (2, ((0, 1.0), (1, 1.0), (2, 1.0), (3, 1.0))),
                (3, ((0, 1.0), (1, 1.0), (2, 1.0), (3, 1.0))),
            ),
        )

        f3 = round(f1, 2)
        self.assertEqual(
            f3.to_pairs(),
            (
                (0, ((0, 0.33), (1, 0.33), (2, 0.33), (3, 0.33))),
                (1, ((0, 0.89), (1, 0.89), (2, 0.89), (3, 0.89))),
                (2, ((0, 0.89), (1, 0.89), (2, 0.89), (3, 0.89))),
                (3, ((0, 0.56), (1, 0.56), (2, 0.56), (3, 0.56))),
            ),
        )

    # ---------------------------------------------------------------------------

    def test_frame_str_capitalize_a(self) -> None:
        f1 = Frame(
            np.array([['foo', 'bar'], ['baz', 'baz']]),
            index=('a', 'b'),
            columns=('x', 'y'),
        )
        f2 = f1.via_str.capitalize()

        self.assertEqual(
            f2.to_pairs(),
            (('x', (('a', 'Foo'), ('b', 'Baz'))), ('y', (('a', 'Bar'), ('b', 'Baz')))),
        )

    def test_frame_str_startswith_a(self) -> None:
        blocks = [
            np.array([['foo', 'bar'], ['baz', 'baz']]),
            np.array(['fall', 'buzz']),
        ]

        f1 = Frame(
            TypeBlocks.from_blocks(blocks), index=('a', 'b'), columns=('x', 'y', 'z')
        )

        f2 = f1.via_str.startswith(('fa', 'ba'))
        self.assertEqual(
            f2.to_pairs(),
            (
                ('x', (('a', False), ('b', True))),
                ('y', (('a', True), ('b', True))),
                ('z', (('a', True), ('b', False))),
            ),
        )

        f3 = f1.via_str.startswith('fo')
        self.assertEqual(
            f3.to_pairs(),
            (
                ('x', (('a', True), ('b', False))),
                ('y', (('a', False), ('b', False))),
                ('z', (('a', False), ('b', False))),
            ),
        )

        f4 = f1.via_str.startswith(('bu', 'fo'))
        self.assertEqual(
            f4.to_pairs(),
            (
                ('x', (('a', True), ('b', False))),
                ('y', (('a', False), ('b', False))),
                ('z', (('a', False), ('b', True))),
            ),
        )

    def test_frame_str_endswith_a(self) -> None:
        blocks = [
            np.array([['foo', 'bar'], ['baz', 'baz']]),
            np.array(['fall', 'buzz']),
        ]

        f1 = Frame(
            TypeBlocks.from_blocks(blocks), index=('a', 'b'), columns=('x', 'y', 'z')
        )

        f2 = f1.via_str.endswith(('zz', 'az'))
        self.assertEqual(
            f1.via_str.endswith(('zz', 'az')).to_pairs(),
            (
                ('x', (('a', False), ('b', True))),
                ('y', (('a', False), ('b', True))),
                ('z', (('a', False), ('b', True))),
            ),
        )

        f3 = f1.via_str.endswith(('oo', 'ar'))
        self.assertEqual(
            f3.to_pairs(),
            (
                ('x', (('a', True), ('b', False))),
                ('y', (('a', True), ('b', False))),
                ('z', (('a', False), ('b', False))),
            ),
        )

    def test_frame_str_center_a(self) -> None:
        f1 = Frame.from_records(
            [['p', 0, True, 'foo'], ['q', 20, None, 'bar']],
            index=('a', 'b'),
            columns=('w', 'x', 'y', 'z'),
        )

        self.assertEqual(
            f1.via_str.center(8, '-').to_pairs(),
            (
                ('w', (('a', '---p----'), ('b', '---q----'))),
                ('x', (('a', '---0----'), ('b', '---20---'))),
                ('y', (('a', '--True--'), ('b', '--None--'))),
                ('z', (('a', '--foo---'), ('b', '--bar---'))),
            ),
        )

    def test_frame_str_partition_a(self) -> None:
        f1 = Frame(
            np.array([['aoc', 'bar'], ['baz', 'baq']]),
            index=('a', 'b'),
            columns=('x', 'y'),
        )
        f2 = f1.via_str.partition('a')
        self.assertEqual(
            f2.to_pairs(),
            (
                ('x', (('a', ('', 'a', 'oc')), ('b', ('b', 'a', 'z')))),
                ('y', (('a', ('b', 'a', 'r')), ('b', ('b', 'a', 'q')))),
            ),
        )

    def test_frame_str_partition_b(self) -> None:
        f1 = Frame(
            np.array([[944, 890], [891, 892]]), index=('a', 'b'), columns=('x', 'y')
        )
        f2 = f1.via_str.partition('9')
        self.assertEqual(
            f2.to_pairs(),
            (
                ('x', (('a', ('', '9', '44')), ('b', ('8', '9', '1')))),
                ('y', (('a', ('8', '9', '0')), ('b', ('8', '9', '2')))),
            ),
        )

    def test_frame_str_islower_a(self) -> None:
        f1 = Frame(
            np.array([['aoc', 'BAR'], ['baz', 'BAQ']]),
            index=('a', 'b'),
            columns=('x', 'y'),
        )
        f2 = f1.via_str.islower()
        self.assertEqual(
            f2.to_pairs(),
            (('x', (('a', True), ('b', True))), ('y', (('a', False), ('b', False)))),
        )

    def test_frame_str_count_a(self) -> None:
        f1 = Frame(
            np.array([['aoc', 'BAR'], ['baz', 'BAQ']]),
            index=('a', 'b'),
            columns=('x', 'y'),
        )
        f2 = f1.via_str.count('BA')
        self.assertEqual(
            f2.to_pairs(), (('x', (('a', 0), ('b', 0))), ('y', (('a', 1), ('b', 1))))
        )

    def test_frame_str_count_b(self) -> None:
        f1 = Frame(np.array([[1, 20], [500, 8332]]), index=('a', 'b'), columns=('x', 'y'))
        f2 = f1.via_str.len()
        self.assertEqual(
            f2.to_pairs(), (('x', (('a', 1), ('b', 3))), ('y', (('a', 2), ('b', 4))))
        )

    # ---------------------------------------------------------------------------

    def test_frame_str_getitem_a(self) -> None:
        f1 = Frame(
            np.array([['foo', 'bar'], ['baz', 'baz']]),
            index=('a', 'b'),
            columns=('x', 'y'),
        )
        f2 = f1.via_str[-1]
        self.assertEqual(
            f2.to_pairs(),
            (('x', (('a', 'o'), ('b', 'z'))), ('y', (('a', 'r'), ('b', 'z')))),
        )

        f3 = f1.via_str[-2:]
        self.assertEqual(
            f3.to_pairs(),
            (('x', (('a', 'oo'), ('b', 'az'))), ('y', (('a', 'ar'), ('b', 'az')))),
        )

    def test_frame_str_getitem_b(self) -> None:
        f1 = Frame(
            np.array([[1004, 1005], [1006, 1007]]), index=('a', 'b'), columns=('x', 'y')
        )
        f2 = f1.via_str[0]
        self.assertEqual(
            f2.to_pairs(),
            (('x', (('a', '1'), ('b', '1'))), ('y', (('a', '1'), ('b', '1')))),
        )

        f3 = f1.via_str[-2:]
        self.assertEqual(
            f3.to_pairs(),
            (('x', (('a', '04'), ('b', '06'))), ('y', (('a', '05'), ('b', '07')))),
        )

    # ---------------------------------------------------------------------------
    def test_frame_str_format_a(self) -> None:
        f1 = Frame(np.array([[10, 20], [30, 40]]), index=('a', 'b'), columns=('x', 'y'))
        f2 = (f1 / 3).via_str.format('{:.3}')
        self.assertEqual(
            f2.to_pairs(),
            (
                ('x', (('a', '3.33'), ('b', '10.0'))),
                ('y', (('a', '6.67'), ('b', '13.3'))),
            ),
        )

    def test_frame_str_format_b(self) -> None:
        f1 = ff.parse('s(2,4)|v(float,float,bool,int)').relabel(columns=list('abcd'))
        f2 = f1.via_str.format(dict(a='{:.2%}', b='{:.2f}', c='b{:=>4}', d='{:b}'))
        self.assertEqual(
            f2.to_pairs(),
            (
                ('a', ((0, '193040.00%'), (1, '-176034.00%'))),
                ('b', ((0, '-610.80'), (1, '3243.94'))),
                ('c', ((0, 'b===1'), (1, 'b===0'))),
                ('d', ((0, '11111011111111001'), (1, '1000100011001101'))),
            ),
        )

    def test_frame_str_format_c(self) -> None:
        f1 = ff.parse('s(2,4)|v(str)').relabel(columns=list('abcd'))
        f2 = f1.via_str.format(dict(c='{:=^12}', d=''))
        self.assertEqual(
            f2.to_pairs(),
            (
                ('a', ((0, 'zjZQ'), (1, 'zO5l'))),
                ('b', ((0, 'zaji'), (1, 'zJnC'))),
                ('c', ((0, '====ztsv===='), (1, '====zUvW===='))),
                ('d', ((0, ''), (1, ''))),
            ),
        )

    def test_frame_str_format_d(self) -> None:
        f1 = ff.parse('s(2,4)|v(str)').relabel(columns=list('abcd'))
        f2 = f1.via_str.format(defaultdict(lambda: '{:=>6}', dict(c='{:=^12}')))
        self.assertEqual(
            f2.to_pairs(),
            (
                ('a', ((0, '==zjZQ'), (1, '==zO5l'))),
                ('b', ((0, '==zaji'), (1, '==zJnC'))),
                ('c', ((0, '====ztsv===='), (1, '====zUvW===='))),
                ('d', ((0, '==z2Oo'), (1, '==z5l6'))),
            ),
        )

    def test_frame_str_format_e(self) -> None:
        f1 = ff.parse('s(2,4)|v(float,float,bool,int)').relabel(columns=IndexAutoFactory)
        f2 = f1.via_str.format({0: '{:.2%}', 1: '{:.2f}', 2: 'b{:=>4}', 3: '{:b}'})
        self.assertEqual(
            f2.to_pairs(),
            (
                (0, ((0, '193040.00%'), (1, '-176034.00%'))),
                (1, ((0, '-610.80'), (1, '3243.94'))),
                (2, ((0, 'b===1'), (1, 'b===0'))),
                (3, ((0, '11111011111111001'), (1, '1000100011001101'))),
            ),
        )

    # ---------------------------------------------------------------------------

    def test_frame_via_dt_year_a(self) -> None:
        dt64 = np.datetime64

        f1 = Frame.from_records(
            [
                ['2012', datetime.date(2012, 4, 5), dt64('2020-05')],
                ['2013', datetime.date(2014, 1, 1), dt64('1919-03')],
            ],
            index=('a', 'b'),
            columns=('w', 'x', 'y'),
        )

        with self.assertRaises(RuntimeError):
            f2 = f1.via_dt.year

        f3 = f1['x':].via_dt.year  # type: ignore
        self.assertEqual(
            f3.to_pairs(),
            (('x', (('a', 2012), ('b', 2014))), ('y', (('a', 2020), ('b', 1919)))),
        )

    def test_frame_via_dt_month_b(self) -> None:
        dt64 = np.datetime64

        f1 = Frame.from_records(
            [
                [
                    datetime.date(2012, 4, 5),
                    datetime.date(2012, 4, 2),
                    dt64('2020-05-03T20:30'),
                    dt64('2017-05-02T05:55'),
                ],
                [
                    datetime.date(2014, 1, 1),
                    datetime.date(2012, 4, 1),
                    dt64('2020-01-03T20:30'),
                    dt64('2025-03-02T03:20'),
                ],
            ],
            index=('a', 'b'),
            columns=('w', 'x', 'y', 'z'),
            consolidate_blocks=True,
        )

        f2 = f1.via_dt.month
        self.assertEqual(
            f2.to_pairs(),
            (
                ('w', (('a', 4), ('b', 1))),
                ('x', (('a', 4), ('b', 4))),
                ('y', (('a', 5), ('b', 1))),
                ('z', (('a', 5), ('b', 3))),
            ),
        )

    def test_frame_via_dt_weekday_a(self) -> None:
        dt64 = np.datetime64

        f1 = Frame.from_records(
            [
                ['2012', datetime.date(2012, 4, 5), dt64('2020-05-03')],
                ['2013', datetime.date(2014, 1, 1), dt64('1919-03-02')],
            ],
            index=('a', 'b'),
            columns=('w', 'x', 'y'),
        )

        with self.assertRaises(RuntimeError):
            f2 = f1.via_dt.weekday()

        self.assertEqual(
            f1['x':].via_dt.weekday().to_pairs(),  # type: ignore
            (('x', (('a', 3), ('b', 2))), ('y', (('a', 6), ('b', 6)))),
        )

    def test_frame_via_dt_weekday_b(self) -> None:
        dt64 = np.datetime64

        f1 = Frame.from_records(
            [
                ['2012', datetime.date(2012, 4, 5), datetime.date(2012, 4, 2)],
                ['2013', datetime.date(2014, 1, 1), datetime.date(2012, 4, 1)],
            ],
            index=('a', 'b'),
            columns=('w', 'x', 'y'),
            consolidate_blocks=True,
        )

        self.assertEqual(
            f1['x':].via_dt.weekday().to_pairs(),  # type: ignore
            (('x', (('a', 3), ('b', 2))), ('y', (('a', 0), ('b', 6)))),
        )

    def test_frame_via_dt_day_a(self) -> None:
        dt64 = np.datetime64

        f1 = Frame.from_records(
            [
                [
                    datetime.date(2012, 4, 5),
                    datetime.date(2012, 4, 2),
                    dt64('2020-05-03T20:30'),
                    dt64('2017-05-02T05:55'),
                ],
                [
                    datetime.date(2014, 1, 1),
                    datetime.date(2012, 4, 1),
                    dt64('2020-01-03T20:30'),
                    dt64('2025-03-02T03:20'),
                ],
            ],
            index=('a', 'b'),
            columns=('w', 'x', 'y', 'z'),
            consolidate_blocks=True,
        )

        f2 = f1.via_dt.day

        self.assertEqual(
            f2.to_pairs(),
            (
                ('w', (('a', 5), ('b', 1))),
                ('x', (('a', 2), ('b', 1))),
                ('y', (('a', 3), ('b', 3))),
                ('z', (('a', 2), ('b', 2))),
            ),
        )

    def test_frame_via_dt_timetuple_a(self) -> None:
        dt64 = np.datetime64

        f1 = Frame.from_records(
            [
                [
                    datetime.date(2012, 4, 5),
                    datetime.date(2012, 4, 2),
                    dt64('2020-05-03T20:30'),
                    dt64('2017-05-02T05:55'),
                ],
                [
                    datetime.date(2014, 1, 1),
                    datetime.date(2012, 4, 1),
                    dt64('2020-01-03T20:30'),
                    dt64('2025-03-02T03:20'),
                ],
            ],
            index=('a', 'b'),
            columns=('w', 'x', 'y', 'z'),
            consolidate_blocks=True,
        )

        import time

        tots = lambda *args: time.struct_time(args)

        self.assertEqual(
            f1.via_dt.timetuple().values.tolist(),
            [
                [
                    tots(2012, 4, 5, 0, 0, 0, 3, 96, -1),
                    tots(2012, 4, 2, 0, 0, 0, 0, 93, -1),
                    tots(2020, 5, 3, 20, 30, 0, 6, 124, -1),
                    tots(2017, 5, 2, 5, 55, 0, 1, 122, -1),
                ],
                [
                    tots(2014, 1, 1, 0, 0, 0, 2, 1, -1),
                    tots(2012, 4, 1, 0, 0, 0, 6, 92, -1),
                    tots(2020, 1, 3, 20, 30, 0, 4, 3, -1),
                    tots(2025, 3, 2, 3, 20, 0, 6, 61, -1),
                ],
            ],  # type: ignore
        )

    def test_frame_via_dt_strftime_a(self) -> None:
        dt64 = np.datetime64

        f1 = Frame.from_records(
            [
                [
                    datetime.date(2012, 4, 5),
                    datetime.date(2012, 4, 2),
                    dt64('2020-05-03T20:30'),
                    dt64('2017-05-02T05:55'),
                ],
                [
                    datetime.date(2014, 1, 1),
                    datetime.date(2012, 4, 1),
                    dt64('2020-01-03T20:30'),
                    dt64('2025-03-02T03:20'),
                ],
            ],
            index=('a', 'b'),
            columns=('w', 'x', 'y', 'z'),
            consolidate_blocks=True,
        )

        f2 = f1.via_dt.strftime('%y|%m|%d')

        self.assertEqual(
            f2.dtypes.values.tolist(),
            [np.dtype('<U8'), np.dtype('<U8'), np.dtype('<U8'), np.dtype('<U8')],
        )

        self.assertEqual(
            f2.to_pairs(),
            (
                ('w', (('a', '12|04|05'), ('b', '14|01|01'))),
                ('x', (('a', '12|04|02'), ('b', '12|04|01'))),
                ('y', (('a', '20|05|03'), ('b', '20|01|03'))),
                ('z', (('a', '17|05|02'), ('b', '25|03|02'))),
            ),
        )

    def test_frame_via_dt_fromisoformat_a(self) -> None:
        dt64 = np.datetime64

        f1 = Frame.from_records(
            [
                [
                    datetime.date(2012, 4, 5),
                    datetime.date(2012, 4, 2),
                    dt64('2020-05-03T20:30'),
                    dt64('2017-05-02T05:55'),
                ],
                [
                    datetime.date(2014, 1, 1),
                    datetime.date(2012, 4, 1),
                    dt64('2020-01-03T20:30'),
                    dt64('2025-03-02T03:20'),
                ],
            ],
            index=('a', 'b'),
            columns=('w', 'x', 'y', 'z'),
            consolidate_blocks=True,
        )

        with self.assertRaises(RuntimeError):
            _ = f1.via_dt.fromisoformat()

    def test_frame_via_dt_fromisoformat_b(self) -> None:
        dt64 = np.datetime64

        f1 = Frame.from_records(
            [
                ['2020-05-03T20:30', ('2017-05-02T05:55')],
                ['2020-01-03T20:30', '2025-03-02T03:20'],
            ],
            index=('a', 'b'),
            columns=('w', 'x'),
            consolidate_blocks=True,
        )

        f2 = f1.via_dt.fromisoformat()

        self.assertEqual(
            f2.to_pairs(),
            (
                (
                    'w',
                    (
                        ('a', datetime.datetime(2020, 5, 3, 20, 30)),
                        ('b', datetime.datetime(2020, 1, 3, 20, 30)),
                    ),
                ),
                (
                    'x',
                    (
                        ('a', datetime.datetime(2017, 5, 2, 5, 55)),
                        ('b', datetime.datetime(2025, 3, 2, 3, 20)),
                    ),
                ),
            ),
        )

    def test_frame_via_dt_strptime_a(self) -> None:
        dt64 = np.datetime64

        f1 = Frame.from_records(
            [['12/1/2012', '3/12/1983'], ['7/3/1972', '12/31/2021']],
            index=('a', 'b'),
            columns=('w', 'x'),
            consolidate_blocks=True,
        )

        f2 = f1.via_dt.strptime('%m/%d/%Y')
        self.assertEqual(
            f2.to_pairs(),
            (
                (
                    'w',
                    (
                        ('a', datetime.datetime(2012, 12, 1, 0, 0)),
                        ('b', datetime.datetime(1972, 7, 3, 0, 0)),
                    ),
                ),
                (
                    'x',
                    (
                        ('a', datetime.datetime(1983, 3, 12, 0, 0)),
                        ('b', datetime.datetime(2021, 12, 31, 0, 0)),
                    ),
                ),
            ),
        )

        f3 = f1.via_dt.strpdate('%m/%d/%Y')
        self.assertEqual(
            f3.to_pairs(),
            (
                (
                    'w',
                    (
                        ('a', datetime.date(2012, 12, 1)),
                        ('b', datetime.date(1972, 7, 3)),
                    ),
                ),
                (
                    'x',
                    (
                        ('a', datetime.date(1983, 3, 12)),
                        ('b', datetime.date(2021, 12, 31)),
                    ),
                ),
            ),
        )

    # ---------------------------------------------------------------------------

    def test_frame_via_dt_year_month_a(self) -> None:
        dt64 = np.datetime64
        f1 = Frame.from_records(
            [
                [
                    datetime.date(2012, 4, 5),
                    datetime.date(2012, 4, 2),
                    dt64('2020-05-03T20:30'),
                    dt64('2017-05-02T05:55'),
                ],
                [
                    datetime.date(2014, 1, 1),
                    datetime.date(2012, 4, 1),
                    dt64('2020-01-03T20:30'),
                    dt64('2025-03-02T03:20'),
                ],
            ],
            index=('a', 'b'),
            columns=('w', 'x', 'y', 'z'),
            consolidate_blocks=True,
        )
        self.assertEqual(
            f1.via_dt.year_month.to_pairs(),
            (
                ('w', (('a', '2012-04'), ('b', '2014-01'))),
                ('x', (('a', '2012-04'), ('b', '2012-04'))),
                ('y', (('a', '2020-05'), ('b', '2020-01'))),
                ('z', (('a', '2017-05'), ('b', '2025-03'))),
            ),
        )

    # ---------------------------------------------------------------------------

    def test_frame_via_dt_year_quarter_a(self) -> None:
        dt64 = np.datetime64
        f1 = Frame.from_records(
            [
                [
                    datetime.date(2012, 4, 5),
                    datetime.date(2012, 4, 2),
                    dt64('2020-05-03T20:30'),
                    dt64('2017-05-02T05:55'),
                ],
                [
                    datetime.date(2014, 1, 1),
                    datetime.date(2012, 4, 1),
                    dt64('2020-01-03T20:30'),
                    dt64('2025-12-02T03:20'),
                ],
            ],
            index=('a', 'b'),
            columns=('w', 'x', 'y', 'z'),
            consolidate_blocks=True,
        )
        self.assertEqual(
            f1.via_dt.year_quarter.to_pairs(),
            (
                ('w', (('a', '2012-Q2'), ('b', '2014-Q1'))),
                ('x', (('a', '2012-Q2'), ('b', '2012-Q2'))),
                ('y', (('a', '2020-Q2'), ('b', '2020-Q1'))),
                ('z', (('a', '2017-Q2'), ('b', '2025-Q4'))),
            ),
        )

    # ---------------------------------------------------------------------------

    def test_frame_via_values_a(self) -> None:
        f = ff.parse('s(3,4)|v(int,float)|c(I,str)')
        # post1 = abs(f).via_values.apply(np.log)

        post1 = np.log(abs(f).via_values)

        self.assertEqual(
            round(post1.fillna(0)).astype(int).to_pairs(),
            (
                ('zZbu', ((0, 11), (1, 11), (2, 11))),
                ('ztsv', ((0, 6), (1, 8), (2, 7))),
                ('zUvW', ((0, 8), (1, 11), (2, 10))),
                ('zkuW', ((0, 7), (1, 8), (2, 7))),
            ),
        )

        post2 = np.log(abs(f).via_values(unify_blocks=True))
        self.assertEqual(
            round(post2.fillna(0)).astype(int).to_pairs(),
            (
                ('zZbu', ((0, 11), (1, 11), (2, 11))),
                ('ztsv', ((0, 6), (1, 8), (2, 7))),
                ('zUvW', ((0, 8), (1, 11), (2, 10))),
                ('zkuW', ((0, 7), (1, 8), (2, 7))),
            ),
        )

        post3 = np.log(abs(f).via_values(consolidate_blocks=True))
        self.assertEqual(
            round(post3.fillna(0)).astype(int).to_pairs(),
            (
                ('zZbu', ((0, 11), (1, 11), (2, 11))),
                ('ztsv', ((0, 6), (1, 8), (2, 7))),
                ('zUvW', ((0, 8), (1, 11), (2, 10))),
                ('zkuW', ((0, 7), (1, 8), (2, 7))),
            ),
        )

        post4 = np.log(abs(f).via_values(consolidate_blocks=True, dtype=float))
        self.assertEqual(
            round(post4.fillna(0)).astype(int).to_pairs(),
            (
                ('zZbu', ((0, 11), (1, 11), (2, 11))),
                ('ztsv', ((0, 6), (1, 8), (2, 7))),
                ('zUvW', ((0, 8), (1, 11), (2, 10))),
                ('zkuW', ((0, 7), (1, 8), (2, 7))),
            ),
        )

        post1 = np.log(abs(f).via_values(dtype=float))
        self.assertEqual(
            round(post1.fillna(0)).astype(int).to_pairs(),
            (
                ('zZbu', ((0, 11), (1, 11), (2, 11))),
                ('ztsv', ((0, 6), (1, 8), (2, 7))),
                ('zUvW', ((0, 8), (1, 11), (2, 10))),
                ('zkuW', ((0, 7), (1, 8), (2, 7))),
            ),
        )

    def test_frame_via_values_b(self) -> None:
        f = ff.parse('s(3,4)|v(bool)|c(I,str)')
        post1 = np.sin(f.via_values)
        self.assertEqual(
            round(post1).to_pairs(),
            (
                ('zZbu', ((0, 0.0), (1, 0.0), (2, 0.0))),
                ('ztsv', ((0, 0.0), (1, 0.0), (2, 0.0))),
                ('zUvW', ((0, 1.0), (1, 0.0), (2, 0.0))),
                ('zkuW', ((0, 0.0), (1, 0.0), (2, 1.0))),
            ),
        )

    def test_frame_via_values_c(self) -> None:
        f = ff.parse('s(3,4)|v(float)|c(I,str)')
        post1 = np.cos(f.via_values)
        self.assertEqual(
            post1.values.round(2).tolist(),
            [
                [0.11, 0.24, -1.0, 0.95],
                [0.5, -0.24, -0.76, -0.46],
                [-0.79, 1.0, -0.73, -0.99],
            ],
        )

    def test_frame_via_values_d(self) -> None:
        f = ff.parse('s(3,4)|v(int)|c(I,str)') % 4
        post1 = np.power(f.via_values, 2)
        self.assertEqual(
            post1.to_pairs(),
            (
                ('zZbu', ((0, 9), (1, 9), (2, 9))),
                ('ztsv', ((0, 1), (1, 9), (2, 1))),
                ('zUvW', ((0, 0), (1, 1), (2, 1))),
                ('zkuW', ((0, 1), (1, 1), (2, 0))),
            ),
        )

        post2 = f.via_values.apply(np.power, 2)
        self.assertEqual(
            post1.to_pairs(),
            (
                ('zZbu', ((0, 9), (1, 9), (2, 9))),
                ('ztsv', ((0, 1), (1, 9), (2, 1))),
                ('zUvW', ((0, 0), (1, 1), (2, 1))),
                ('zkuW', ((0, 1), (1, 1), (2, 0))),
            ),
        )

    def test_frame_via_values_e(self) -> None:
        f = ff.parse('s(3,4)|v(int, bool)|c(I,str)')

        post1 = np.power(f.via_values(), 2)
        self.assertEqual([dt.kind for dt in post1.dtypes.values], ['i', 'i', 'i', 'i'])

        post2 = np.power(f.via_values(unify_blocks=True), 2)
        self.assertEqual([dt.kind for dt in post2.dtypes.values], ['O', 'O', 'O', 'O'])

    # ---------------------------------------------------------------------------

    def test_frame_equals_a(self) -> None:
        idx1 = IndexHierarchy.from_product(
            ('far', 20), (None, 'down'), (False, 'right'), name=('z', 'y', 'x')
        )
        idx2 = IndexHierarchy.from_product(
            ('far', 20), (None, 'down'), (False, 'right'), name=('z', 'y', 'x')
        )
        f1 = FrameGO(np.arange(16, dtype=np.int64).reshape(8, 2), index=idx1)
        f2 = FrameGO(np.arange(16, dtype=np.int64).reshape(8, 2), index=idx2)
        f3 = FrameGO(np.arange(16, dtype=np.int64).reshape(8, 2), index=idx2, name='foo')
        f4 = FrameGO(np.arange(16, dtype=np.int32).reshape(8, 2), index=idx2)
        f5 = Frame(np.arange(16, dtype=np.int64).reshape(8, 2), index=idx2)

        self.assertTrue(f1.equals(f1))
        self.assertTrue(f1.equals(f2))

        self.assertFalse(f1.equals(f3, compare_name=True))
        self.assertTrue(f1.equals(f3, compare_name=False))

        self.assertFalse(f1.equals(f4, compare_dtype=True))
        self.assertTrue(f1.equals(f4, compare_dtype=False))

        self.assertFalse(f1.equals(f5, compare_class=True))
        self.assertTrue(f1.equals(f5, compare_class=False))

    def test_frame_equals_b(self) -> None:
        idx1 = IndexHierarchy.from_product(
            ('far', 20), (None, 'down'), (False, 'right'), name=('z', 'y', 'x')
        )
        idx2 = IndexHierarchy.from_product(
            ('far', 20), (None, 'down'), (False, 'right'), name=('z', 'y', 'q')
        )
        f1 = FrameGO(np.arange(16, dtype=np.int64).reshape(8, 2), index=idx1)
        f2 = FrameGO(np.arange(16, dtype=np.int64).reshape(8, 2), index=idx2)

        self.assertFalse(f1.equals(f2, compare_name=True))
        self.assertTrue(f1.equals(f2, compare_name=False))

    def test_frame_equals_c(self) -> None:
        idx1 = IndexHierarchy.from_product(
            ('far', 20), (None, 'down'), (False, 'right'), name=('z', 'y', 'x')
        )
        idx2 = IndexHierarchy.from_product(
            ('far', 20), (None, 'down'), (False, 'right'), name=('z', 'y', 'x')
        )
        f1 = FrameGO(np.arange(16, dtype=np.int64).reshape(8, 2), index=idx1)
        f2 = FrameGO(
            np.arange(16, dtype=np.int64).reshape(8, 2), index=idx2, columns=('a', 'b')
        )
        f3 = FrameGO(np.arange(24, dtype=np.int64).reshape(8, 3), index=idx2)

        self.assertFalse(f1.equals(f2))
        self.assertFalse(f1.equals(f3))

    def test_frame_equals_d(self) -> None:
        records = (
            (2, 2),
            (np.nan, 0),
            (np.nan, -95),
        )
        f1 = Frame.from_records(records, columns=('a', 'b'), index=('x', 'y', 'z'))

        f2 = Frame.from_records(records, columns=('a', 'b'), index=('x', 'y', 'z'))

        self.assertTrue(f1.equals(f2))
        self.assertFalse(f1.equals(f2, skipna=False))

    def test_frame_equals_e(self) -> None:
        records = (
            (2, 2),
            (3, 0),
            (5, -95),
        )
        f1 = Frame.from_records(records, columns=('a', 'b'), index=('x', 'y', np.nan))

        f2 = Frame.from_records(records, columns=('a', 'b'), index=('x', 'y', np.nan))

        self.assertTrue(f1.equals(f2))
        self.assertFalse(f1.equals(f2, skipna=False))

    def test_frame_equals_f(self) -> None:
        f1 = Frame.from_element('a', index=range(2), columns=range(2))
        f2 = Frame.from_element(3, index=range(2), columns=range(2))

        self.assertFalse(f1.equals(f2, compare_dtype=False))

    def test_frame_equals_g(self) -> None:
        f1 = Frame.from_element('a', index=range(2), columns=range(2))
        self.assertFalse(f1.equals(f1.values, compare_class=True))
        self.assertFalse(f1.equals(f1.values, compare_class=False))

    # ---------------------------------------------------------------------------
    def test_frame_append_a(self) -> None:
        f1 = FrameGO(
            index=('a', 'b'),
            columns=IndexHierarchyGO.from_labels((), depth_reference=3),
        )
        f1[('a', 1, True)] = 30

        self.assertEqual(f1.to_pairs(), ((('a', 1, True), (('a', 30), ('b', 30))),))

    def test_frame_append_b(self) -> None:
        f1 = FrameGO(columns=IndexHierarchyGO.from_names(('foo', 'bar')), index=range(2))
        f1[('A', 1)] = 10
        f1[('A', 2)] = 20
        f1[('B', 2)] = 30

        self.assertEqual(
            f1.to_pairs(),
            (
                (('A', 1), ((0, 10), (1, 10))),
                (('A', 2), ((0, 20), (1, 20))),
                (('B', 2), ((0, 30), (1, 30))),
            ),
        )

    # ---------------------------------------------------------------------------

    def test_frame_pivot_stack_a(self) -> None:
        f1 = Frame.from_records(
            [[0, 'w'], [1, 'x'], [2, 'y'], [3, 'z']],
            columns=('a', 'b'),
            index=IndexHierarchy.from_product(('I', 'II'), (1, 2), name='foo'),
            name='bar',
        )
        f2 = f1.pivot_stack()
        self.assertEqual(
            f2.to_pairs(),
            (
                (
                    0,
                    (
                        (('I', 1, 'a'), 0),
                        (('I', 1, 'b'), 'w'),
                        (('I', 2, 'a'), 1),
                        (('I', 2, 'b'), 'x'),
                        (('II', 1, 'a'), 2),
                        (('II', 1, 'b'), 'y'),
                        (('II', 2, 'a'), 3),
                        (('II', 2, 'b'), 'z'),
                    ),
                ),
            ),
        )

        self.assertEqual(f2.index.name, None)
        self.assertEqual(f2.name, 'bar')

    def test_frame_pivot_stack_b(self) -> None:
        f1 = Frame.from_records(
            np.arange(16).reshape(8, 2),
            columns=('a', 'b'),
            index=IndexHierarchy.from_product(
                ('foo', 'bar'), IndexDate.from_date_range('2000-01-01', '2000-01-04')
            ),
        )

        f2 = f1.pivot_stack()

        self.assertEqual(f2.index.index_types.values.tolist(), [Index, IndexDate, Index])

        self.assertEqual(
            f2.to_pairs(),
            (
                (
                    0,
                    (
                        (('foo', datetime.date(2000, 1, 1), 'a'), 0),
                        (('foo', datetime.date(2000, 1, 1), 'b'), 1),
                        (('foo', datetime.date(2000, 1, 2), 'a'), 2),
                        (('foo', datetime.date(2000, 1, 2), 'b'), 3),
                        (('foo', datetime.date(2000, 1, 3), 'a'), 4),
                        (('foo', datetime.date(2000, 1, 3), 'b'), 5),
                        (('foo', datetime.date(2000, 1, 4), 'a'), 6),
                        (('foo', datetime.date(2000, 1, 4), 'b'), 7),
                        (('bar', datetime.date(2000, 1, 1), 'a'), 8),
                        (('bar', datetime.date(2000, 1, 1), 'b'), 9),
                        (('bar', datetime.date(2000, 1, 2), 'a'), 10),
                        (('bar', datetime.date(2000, 1, 2), 'b'), 11),
                        (('bar', datetime.date(2000, 1, 3), 'a'), 12),
                        (('bar', datetime.date(2000, 1, 3), 'b'), 13),
                        (('bar', datetime.date(2000, 1, 4), 'a'), 14),
                        (('bar', datetime.date(2000, 1, 4), 'b'), 15),
                    ),
                ),
            ),
        )

    def test_frame_pivot_stack_c(self) -> None:
        index = IndexHierarchy.from_labels((('r0', 'r00'), ('r0', 'r01')))
        columns = IndexHierarchy.from_labels(
            (('c0', 'c00'), ('c0', 'c01'), ('c1', 'c10'))
        )
        f1 = Frame(np.arange(6).reshape(2, 3), index=index, columns=columns)

        f2 = f1.pivot_stack(fill_value=-1)

        self.assertEqual(
            f2.to_pairs(),
            (
                (
                    'c0',
                    (
                        (('r0', 'r00', 'c00'), 0),
                        (('r0', 'r00', 'c01'), 1),
                        (('r0', 'r00', 'c10'), -1),
                        (('r0', 'r01', 'c00'), 3),
                        (('r0', 'r01', 'c01'), 4),
                        (('r0', 'r01', 'c10'), -1),
                    ),
                ),
                (
                    'c1',
                    (
                        (('r0', 'r00', 'c00'), -1),
                        (('r0', 'r00', 'c01'), -1),
                        (('r0', 'r00', 'c10'), 2),
                        (('r0', 'r01', 'c00'), -1),
                        (('r0', 'r01', 'c01'), -1),
                        (('r0', 'r01', 'c10'), 5),
                    ),
                ),
            ),
        )

    def test_frame_pivot_stack_d(self) -> None:
        for cls in (Frame, FrameGO):
            f1 = cls(
                np.arange(16).reshape(2, 8),
                columns=IndexHierarchy.from_product(('I', 'II'), ('A', 'B'), (1, 2)),
            )

            f2 = f1.pivot_stack()

            self.assertEqual(
                f2.to_pairs(),
                (
                    (('I', 'A'), (((0, 1), 0), ((0, 2), 1), ((1, 1), 8), ((1, 2), 9))),
                    (
                        ('I', 'B'),
                        (((0, 1), 2), ((0, 2), 3), ((1, 1), 10), ((1, 2), 11)),
                    ),
                    (
                        ('II', 'A'),
                        (((0, 1), 4), ((0, 2), 5), ((1, 1), 12), ((1, 2), 13)),
                    ),
                    (
                        ('II', 'B'),
                        (((0, 1), 6), ((0, 2), 7), ((1, 1), 14), ((1, 2), 15)),
                    ),
                ),
            )

            f3 = f2.pivot_stack()

            self.assertEqual(
                f3.to_pairs(),
                (
                    (
                        'I',
                        (
                            ((0, 1, 'A'), 0),
                            ((0, 1, 'B'), 2),
                            ((0, 2, 'A'), 1),
                            ((0, 2, 'B'), 3),
                            ((1, 1, 'A'), 8),
                            ((1, 1, 'B'), 10),
                            ((1, 2, 'A'), 9),
                            ((1, 2, 'B'), 11),
                        ),
                    ),
                    (
                        'II',
                        (
                            ((0, 1, 'A'), 4),
                            ((0, 1, 'B'), 6),
                            ((0, 2, 'A'), 5),
                            ((0, 2, 'B'), 7),
                            ((1, 1, 'A'), 12),
                            ((1, 1, 'B'), 14),
                            ((1, 2, 'A'), 13),
                            ((1, 2, 'B'), 15),
                        ),
                    ),
                ),
            )

            f4 = f3.pivot_stack()

            self.assertEqual(
                f4.to_pairs(),
                (
                    (
                        0,
                        (
                            ((0, 1, 'A', 'I'), 0),
                            ((0, 1, 'A', 'II'), 4),
                            ((0, 1, 'B', 'I'), 2),
                            ((0, 1, 'B', 'II'), 6),
                            ((0, 2, 'A', 'I'), 1),
                            ((0, 2, 'A', 'II'), 5),
                            ((0, 2, 'B', 'I'), 3),
                            ((0, 2, 'B', 'II'), 7),
                            ((1, 1, 'A', 'I'), 8),
                            ((1, 1, 'A', 'II'), 12),
                            ((1, 1, 'B', 'I'), 10),
                            ((1, 1, 'B', 'II'), 14),
                            ((1, 2, 'A', 'I'), 9),
                            ((1, 2, 'A', 'II'), 13),
                            ((1, 2, 'B', 'I'), 11),
                            ((1, 2, 'B', 'II'), 15),
                        ),
                    ),
                ),
            )

            f5 = f1.pivot_stack(0)  # the outermost
            self.assertEqual(
                f5.to_pairs(),
                (
                    (
                        ('A', 1),
                        (((0, 'I'), 0), ((0, 'II'), 4), ((1, 'I'), 8), ((1, 'II'), 12)),
                    ),
                    (
                        ('A', 2),
                        (((0, 'I'), 1), ((0, 'II'), 5), ((1, 'I'), 9), ((1, 'II'), 13)),
                    ),
                    (
                        ('B', 1),
                        (
                            ((0, 'I'), 2),
                            ((0, 'II'), 6),
                            ((1, 'I'), 10),
                            ((1, 'II'), 14),
                        ),
                    ),
                    (
                        ('B', 2),
                        (
                            ((0, 'I'), 3),
                            ((0, 'II'), 7),
                            ((1, 'I'), 11),
                            ((1, 'II'), 15),
                        ),
                    ),
                ),
            )

            f6 = f1.pivot_stack(1)
            self.assertEqual(
                f6.to_pairs(),
                (
                    (
                        ('I', 1),
                        (((0, 'A'), 0), ((0, 'B'), 2), ((1, 'A'), 8), ((1, 'B'), 10)),
                    ),
                    (
                        ('I', 2),
                        (((0, 'A'), 1), ((0, 'B'), 3), ((1, 'A'), 9), ((1, 'B'), 11)),
                    ),
                    (
                        ('II', 1),
                        (((0, 'A'), 4), ((0, 'B'), 6), ((1, 'A'), 12), ((1, 'B'), 14)),
                    ),
                    (
                        ('II', 2),
                        (((0, 'A'), 5), ((0, 'B'), 7), ((1, 'A'), 13), ((1, 'B'), 15)),
                    ),
                ),
            )

    def test_frame_pivot_stack_e1(self) -> None:
        f1 = Frame(
            np.arange(16).reshape(2, 8),
            columns=IndexHierarchy.from_product(('I', 'II'), ('A', 'B'), (1, 2)),
        )

        f2 = f1.pivot_stack([0, 2])
        self.assertEqual(
            f2.to_pairs(),
            (
                (
                    'A',
                    (
                        ((0, 'I', 1), 0),
                        ((0, 'I', 2), 1),
                        ((0, 'II', 1), 4),
                        ((0, 'II', 2), 5),
                        ((1, 'I', 1), 8),
                        ((1, 'I', 2), 9),
                        ((1, 'II', 1), 12),
                        ((1, 'II', 2), 13),
                    ),
                ),
                (
                    'B',
                    (
                        ((0, 'I', 1), 2),
                        ((0, 'I', 2), 3),
                        ((0, 'II', 1), 6),
                        ((0, 'II', 2), 7),
                        ((1, 'I', 1), 10),
                        ((1, 'I', 2), 11),
                        ((1, 'II', 1), 14),
                        ((1, 'II', 2), 15),
                    ),
                ),
            ),
        )

        f3 = f1.pivot_stack([1, 2])
        self.assertEqual(
            f3.to_pairs(),
            (
                (
                    'I',
                    (
                        ((0, 'A', 1), 0),
                        ((0, 'A', 2), 1),
                        ((0, 'B', 1), 2),
                        ((0, 'B', 2), 3),
                        ((1, 'A', 1), 8),
                        ((1, 'A', 2), 9),
                        ((1, 'B', 1), 10),
                        ((1, 'B', 2), 11),
                    ),
                ),
                (
                    'II',
                    (
                        ((0, 'A', 1), 4),
                        ((0, 'A', 2), 5),
                        ((0, 'B', 1), 6),
                        ((0, 'B', 2), 7),
                        ((1, 'A', 1), 12),
                        ((1, 'A', 2), 13),
                        ((1, 'B', 1), 14),
                        ((1, 'B', 2), 15),
                    ),
                ),
            ),
        )

    def test_frame_pivot_stack_e2(self) -> None:
        f1 = Frame(
            np.arange(16).reshape(2, 8),
            columns=IndexHierarchy.from_product(('I', 'II'), ('A', 'B'), (1, 2)),
        )
        with self.assertRaises(InvalidFillValue):
            f2 = f1.pivot_stack([0, 2], fill_value=FillValueAuto)

    def test_frame_pivot_stack_f(self) -> None:
        f1 = Frame.from_records(
            np.arange(16).reshape(4, 4),
            columns=IndexHierarchy.from_product(
                IndexYear(('1642', '1633')),
                IndexDate.from_date_range('1733-01-01', '1733-01-02'),
            ),
            index=IndexHierarchy.from_product(
                IndexYear(('1810', '1840')),
                IndexDate.from_date_range('2000-01-01', '2000-01-02'),
            ),
        )

        f2 = f1.pivot_unstack()

        self.assertEqual(
            f2.columns.index_types.values.tolist(), [IndexYear, IndexDate, IndexDate]
        )
        self.assertEqual(f2.index.__class__, IndexYear)

        dt64 = np.datetime64
        self.assertEqual(
            f2.to_pairs(),
            (
                (
                    (dt64('1642-01-01'), dt64('1733-01-01'), dt64('2000-01-01')),
                    ((dt64('1810'), 0), (dt64('1840'), 8)),
                ),
                (
                    (dt64('1642-01-01'), dt64('1733-01-01'), dt64('2000-01-02')),
                    ((dt64('1810'), 4), (dt64('1840'), 12)),
                ),
                (
                    (dt64('1642-01-01'), dt64('1733-01-02'), dt64('2000-01-01')),
                    ((dt64('1810'), 1), (dt64('1840'), 9)),
                ),
                (
                    (dt64('1642-01-01'), dt64('1733-01-02'), dt64('2000-01-02')),
                    ((dt64('1810'), 5), (dt64('1840'), 13)),
                ),
                (
                    (dt64('1633-01-01'), dt64('1733-01-01'), dt64('2000-01-01')),
                    ((dt64('1810'), 2), (dt64('1840'), 10)),
                ),
                (
                    (dt64('1633-01-01'), dt64('1733-01-01'), dt64('2000-01-02')),
                    ((dt64('1810'), 6), (dt64('1840'), 14)),
                ),
                (
                    (dt64('1633-01-01'), dt64('1733-01-02'), dt64('2000-01-01')),
                    ((dt64('1810'), 3), (dt64('1840'), 11)),
                ),
                (
                    (dt64('1633-01-01'), dt64('1733-01-02'), dt64('2000-01-02')),
                    ((dt64('1810'), 7), (dt64('1840'), 15)),
                ),
            ),
        )

        f3 = f1.pivot_stack()
        self.assertEqual(
            f3.index.index_types.values.tolist(), [IndexYear, IndexDate, IndexDate]
        )
        self.assertEqual(f3.columns.__class__, IndexYear)

        self.assertEqual(
            f3.to_pairs(),
            (
                (
                    dt64('1642'),
                    (
                        (
                            (
                                dt64('1810-01-01'),
                                dt64('2000-01-01'),
                                dt64('1733-01-01'),
                            ),
                            0,
                        ),
                        (
                            (
                                dt64('1810-01-01'),
                                dt64('2000-01-01'),
                                dt64('1733-01-02'),
                            ),
                            1,
                        ),
                        (
                            (
                                dt64('1810-01-01'),
                                dt64('2000-01-02'),
                                dt64('1733-01-01'),
                            ),
                            4,
                        ),
                        (
                            (
                                dt64('1810-01-01'),
                                dt64('2000-01-02'),
                                dt64('1733-01-02'),
                            ),
                            5,
                        ),
                        (
                            (
                                dt64('1840-01-01'),
                                dt64('2000-01-01'),
                                dt64('1733-01-01'),
                            ),
                            8,
                        ),
                        (
                            (
                                dt64('1840-01-01'),
                                dt64('2000-01-01'),
                                dt64('1733-01-02'),
                            ),
                            9,
                        ),
                        (
                            (
                                dt64('1840-01-01'),
                                dt64('2000-01-02'),
                                dt64('1733-01-01'),
                            ),
                            12,
                        ),
                        (
                            (
                                dt64('1840-01-01'),
                                dt64('2000-01-02'),
                                dt64('1733-01-02'),
                            ),
                            13,
                        ),
                    ),
                ),
                (
                    dt64('1633'),
                    (
                        (
                            (
                                dt64('1810-01-01'),
                                dt64('2000-01-01'),
                                dt64('1733-01-01'),
                            ),
                            2,
                        ),
                        (
                            (
                                dt64('1810-01-01'),
                                dt64('2000-01-01'),
                                dt64('1733-01-02'),
                            ),
                            3,
                        ),
                        (
                            (
                                dt64('1810-01-01'),
                                dt64('2000-01-02'),
                                dt64('1733-01-01'),
                            ),
                            6,
                        ),
                        (
                            (
                                dt64('1810-01-01'),
                                dt64('2000-01-02'),
                                dt64('1733-01-02'),
                            ),
                            7,
                        ),
                        (
                            (
                                dt64('1840-01-01'),
                                dt64('2000-01-01'),
                                dt64('1733-01-01'),
                            ),
                            10,
                        ),
                        (
                            (
                                dt64('1840-01-01'),
                                dt64('2000-01-01'),
                                dt64('1733-01-02'),
                            ),
                            11,
                        ),
                        (
                            (
                                dt64('1840-01-01'),
                                dt64('2000-01-02'),
                                dt64('1733-01-01'),
                            ),
                            14,
                        ),
                        (
                            (
                                dt64('1840-01-01'),
                                dt64('2000-01-02'),
                                dt64('1733-01-02'),
                            ),
                            15,
                        ),
                    ),
                ),
            ),
        )

    def test_frame_pivot_stack_g(self) -> None:
        f1 = FrameGO.from_records(
            np.arange(16).reshape(4, 4),
            columns=IndexHierarchy.from_product(
                IndexYear(('1642', '1633')),
                IndexDate.from_date_range('1733-01-01', '1733-01-02'),
            ),
            index=IndexHierarchy.from_product(
                IndexYear(('1810', '1840')),
                IndexDate.from_date_range('2000-01-01', '2000-01-02'),
            ),
        )

        f2 = f1.pivot_unstack()

        self.assertEqual(
            f2.columns.index_types.values.tolist(),
            [IndexYearGO, IndexDateGO, IndexDateGO],
        )
        self.assertEqual(f2.index.__class__, IndexYear)

        dt64 = np.datetime64
        self.assertEqual(
            f2.to_pairs(),
            (
                (
                    (dt64('1642-01-01'), dt64('1733-01-01'), dt64('2000-01-01')),
                    ((dt64('1810'), 0), (dt64('1840'), 8)),
                ),
                (
                    (dt64('1642-01-01'), dt64('1733-01-01'), dt64('2000-01-02')),
                    ((dt64('1810'), 4), (dt64('1840'), 12)),
                ),
                (
                    (dt64('1642-01-01'), dt64('1733-01-02'), dt64('2000-01-01')),
                    ((dt64('1810'), 1), (dt64('1840'), 9)),
                ),
                (
                    (dt64('1642-01-01'), dt64('1733-01-02'), dt64('2000-01-02')),
                    ((dt64('1810'), 5), (dt64('1840'), 13)),
                ),
                (
                    (dt64('1633-01-01'), dt64('1733-01-01'), dt64('2000-01-01')),
                    ((dt64('1810'), 2), (dt64('1840'), 10)),
                ),
                (
                    (dt64('1633-01-01'), dt64('1733-01-01'), dt64('2000-01-02')),
                    ((dt64('1810'), 6), (dt64('1840'), 14)),
                ),
                (
                    (dt64('1633-01-01'), dt64('1733-01-02'), dt64('2000-01-01')),
                    ((dt64('1810'), 3), (dt64('1840'), 11)),
                ),
                (
                    (dt64('1633-01-01'), dt64('1733-01-02'), dt64('2000-01-02')),
                    ((dt64('1810'), 7), (dt64('1840'), 15)),
                ),
            ),
        )

        f3 = f1.pivot_stack()
        self.assertEqual(
            f3.index.index_types.values.tolist(), [IndexYear, IndexDate, IndexDate]
        )
        self.assertEqual(f3.columns.__class__, IndexYearGO)

        self.assertEqual(
            f3.to_pairs(),
            (
                (
                    dt64('1642'),
                    (
                        (
                            (
                                dt64('1810-01-01'),
                                dt64('2000-01-01'),
                                dt64('1733-01-01'),
                            ),
                            0,
                        ),
                        (
                            (
                                dt64('1810-01-01'),
                                dt64('2000-01-01'),
                                dt64('1733-01-02'),
                            ),
                            1,
                        ),
                        (
                            (
                                dt64('1810-01-01'),
                                dt64('2000-01-02'),
                                dt64('1733-01-01'),
                            ),
                            4,
                        ),
                        (
                            (
                                dt64('1810-01-01'),
                                dt64('2000-01-02'),
                                dt64('1733-01-02'),
                            ),
                            5,
                        ),
                        (
                            (
                                dt64('1840-01-01'),
                                dt64('2000-01-01'),
                                dt64('1733-01-01'),
                            ),
                            8,
                        ),
                        (
                            (
                                dt64('1840-01-01'),
                                dt64('2000-01-01'),
                                dt64('1733-01-02'),
                            ),
                            9,
                        ),
                        (
                            (
                                dt64('1840-01-01'),
                                dt64('2000-01-02'),
                                dt64('1733-01-01'),
                            ),
                            12,
                        ),
                        (
                            (
                                dt64('1840-01-01'),
                                dt64('2000-01-02'),
                                dt64('1733-01-02'),
                            ),
                            13,
                        ),
                    ),
                ),
                (
                    dt64('1633'),
                    (
                        (
                            (
                                dt64('1810-01-01'),
                                dt64('2000-01-01'),
                                dt64('1733-01-01'),
                            ),
                            2,
                        ),
                        (
                            (
                                dt64('1810-01-01'),
                                dt64('2000-01-01'),
                                dt64('1733-01-02'),
                            ),
                            3,
                        ),
                        (
                            (
                                dt64('1810-01-01'),
                                dt64('2000-01-02'),
                                dt64('1733-01-01'),
                            ),
                            6,
                        ),
                        (
                            (
                                dt64('1810-01-01'),
                                dt64('2000-01-02'),
                                dt64('1733-01-02'),
                            ),
                            7,
                        ),
                        (
                            (
                                dt64('1840-01-01'),
                                dt64('2000-01-01'),
                                dt64('1733-01-01'),
                            ),
                            10,
                        ),
                        (
                            (
                                dt64('1840-01-01'),
                                dt64('2000-01-01'),
                                dt64('1733-01-02'),
                            ),
                            11,
                        ),
                        (
                            (
                                dt64('1840-01-01'),
                                dt64('2000-01-02'),
                                dt64('1733-01-01'),
                            ),
                            14,
                        ),
                        (
                            (
                                dt64('1840-01-01'),
                                dt64('2000-01-02'),
                                dt64('1733-01-02'),
                            ),
                            15,
                        ),
                    ),
                ),
            ),
        )

    # ---------------------------------------------------------------------------

    def test_frame_pivot_unstack_a1(self) -> None:
        index = IndexHierarchy.from_labels((('r0', 'r00'), ('r0', 'r01')))
        columns = IndexHierarchy.from_labels(
            (('c0', 'c00'), ('c0', 'c01'), ('c1', 'c10'))
        )
        f1 = Frame(np.arange(6).reshape(2, 3), index=index, columns=columns)

        f2 = f1.pivot_unstack(fill_value=-1)

        self.assertEqual(
            f2.to_pairs(),
            (
                (('c0', 'c00', 'r00'), (('r0', 0),)),
                (('c0', 'c00', 'r01'), (('r0', 3),)),
                (('c0', 'c01', 'r00'), (('r0', 1),)),
                (('c0', 'c01', 'r01'), (('r0', 4),)),
                (('c1', 'c10', 'r00'), (('r0', 2),)),
                (('c1', 'c10', 'r01'), (('r0', 5),)),
            ),
        )

        f3 = f2.pivot_unstack()

        self.assertEqual(
            f3.to_pairs(),
            (
                (('c0', 'c00', 'r00', 'r0'), ((0, 0),)),
                (('c0', 'c00', 'r01', 'r0'), ((0, 3),)),
                (('c0', 'c01', 'r00', 'r0'), ((0, 1),)),
                (('c0', 'c01', 'r01', 'r0'), ((0, 4),)),
                (('c1', 'c10', 'r00', 'r0'), ((0, 2),)),
                (('c1', 'c10', 'r01', 'r0'), ((0, 5),)),
            ),
        )

    def test_frame_pivot_unstack_a2(self) -> None:
        index = IndexHierarchy.from_labels((('r0', 'r00'), ('r0', 'r01')))
        columns = IndexHierarchy.from_labels(
            (('c0', 'c00'), ('c0', 'c01'), ('c1', 'c10'))
        )
        f1 = Frame(np.arange(6).reshape(2, 3), index=index, columns=columns)
        with self.assertRaises(InvalidFillValue):
            _ = f1.pivot_unstack(fill_value=FillValueAuto)

    def test_frame_pivot_unstack_b(self) -> None:
        f1 = sf.Frame.from_records(
            (
                ('muon', 0.106, -1.0, 'lepton'),
                ('tau', 1.777, -1.0, 'lepton'),
                ('charm', 1.3, 0.666, 'quark'),
                ('strange', 0.1, -0.333, 'quark'),
            ),
            columns=('name', 'mass', 'charge', 'type'),
        )

        f1 = f1.set_index_hierarchy(('type', 'name'), drop=True)
        f2 = f1.pivot_unstack([0, 1])

        self.assertEqual(
            f2.to_pairs(),
            (
                (('mass', 'lepton', 'muon'), ((0, 0.106),)),
                (('mass', 'lepton', 'tau'), ((0, 1.777),)),
                (('mass', 'quark', 'charm'), ((0, 1.3),)),
                (('mass', 'quark', 'strange'), ((0, 0.1),)),
                (('charge', 'lepton', 'muon'), ((0, -1.0),)),
                (('charge', 'lepton', 'tau'), ((0, -1.0),)),
                (('charge', 'quark', 'charm'), ((0, 0.666),)),
                (('charge', 'quark', 'strange'), ((0, -0.333),)),
            ),
        )

    def test_frame_pivot_unstack_c(self) -> None:
        f1 = sf.Frame.from_records(
            (
                ('muon', 0.106, -1.0, 'lepton'),
                ('tau', 1.777, -1.0, 'lepton'),
                ('charm', 1.3, 0.666, 'quark'),
                ('strange', 0.1, -0.333, 'quark'),
            ),
            columns=('name', 'mass', 'charge', 'type'),
        )

        f1 = f1.set_index_hierarchy(('type', 'name'), drop=True)
        f2 = f1.pivot_unstack(0)
        self.assertEqual(
            f2.fillna(0).to_pairs(),
            (
                (
                    ('mass', 'lepton'),
                    (('muon', 0.106), ('tau', 1.777), ('charm', 0.0), ('strange', 0.0)),
                ),
                (
                    ('mass', 'quark'),
                    (('muon', 0.0), ('tau', 0.0), ('charm', 1.3), ('strange', 0.1)),
                ),
                (
                    ('charge', 'lepton'),
                    (('muon', -1.0), ('tau', -1.0), ('charm', 0.0), ('strange', 0.0)),
                ),
                (
                    ('charge', 'quark'),
                    (
                        ('muon', 0.0),
                        ('tau', 0.0),
                        ('charm', 0.666),
                        ('strange', -0.333),
                    ),
                ),
            ),
        )

        f3 = f1.pivot_unstack(1)
        self.assertEqual(
            f3.fillna(0).to_pairs(),
            (
                (('mass', 'muon'), (('lepton', 0.106), ('quark', 0.0))),
                (('mass', 'tau'), (('lepton', 1.777), ('quark', 0.0))),
                (('mass', 'charm'), (('lepton', 0.0), ('quark', 1.3))),
                (('mass', 'strange'), (('lepton', 0.0), ('quark', 0.1))),
                (('charge', 'muon'), (('lepton', -1.0), ('quark', 0.0))),
                (('charge', 'tau'), (('lepton', -1.0), ('quark', 0.0))),
                (('charge', 'charm'), (('lepton', 0.0), ('quark', 0.666))),
                (('charge', 'strange'), (('lepton', 0.0), ('quark', -0.333))),
            ),
        )

    # ---------------------------------------------------------------------------

    def test_frame_from_overlay_a(self) -> None:
        f1 = Frame.from_dict(dict(a=(1, 2), b=(3, 4)), index=('x', 'y'), name='f1')
        f2 = Frame.from_dict(
            dict(c=(1, 2, 3), b=(4, 5, 6)), index=('x', 'y', 'z'), name='f2'
        )
        f3 = Frame.from_dict(dict(d=(10, 20), b=(50, 60)), index=('x', 'q'), name='f3')

        f4 = sf.Frame.from_overlay((f1, f2, f3), name='foo')
        self.assertEqual(
            f4.fillna(-1).to_pairs(),
            (
                ('a', (('q', -1.0), ('x', 1.0), ('y', 2.0), ('z', -1.0))),
                ('b', (('q', 60.0), ('x', 3.0), ('y', 4.0), ('z', 6.0))),
                ('c', (('q', -1.0), ('x', 1.0), ('y', 2.0), ('z', 3.0))),
                ('d', (('q', 20.0), ('x', 10.0), ('y', -1.0), ('z', -1.0))),
            ),
        )
        self.assertEqual(f4.name, 'foo')

    def test_frame_from_overlay_b(self) -> None:
        f1 = Frame.from_dict(
            dict(a=(1, np.nan), b=(np.nan, np.nan), c=(False, True)),
            index=('x', 'y'),
            name='f1',
        )
        f2 = Frame.from_dict(
            dict(a=(1, np.nan), b=(4, 6), c=(False, True)), index=('x', 'y'), name='f2'
        )
        f3 = Frame.from_dict(
            dict(a=(10, 20), b=(50, 60), c=(False, True)), index=('x', 'y'), name='f3'
        )

        f4 = sf.Frame.from_overlay((f1, f2, f3))

        self.assertEqual(
            f4.to_pairs(),
            (
                ('a', (('x', 1.0), ('y', 20))),
                ('b', (('x', 4), ('y', 6))),
                ('c', (('x', False), ('y', True))),
            ),
        )

        self.assertEqual([dt.kind for dt in f4.dtypes.values], ['f', 'i', 'b'])

    def test_frame_from_overlay_c(self) -> None:
        f1 = Frame.from_dict(
            dict(a=(1, np.nan), b=(np.nan, 20), c=(False, None)),
            index=Index(('x', 'y'), name='foo'),
            name='f1',
        )
        f2 = Frame.from_dict(
            dict(a=(1, 200), b=(4, 6), c=(False, False)),
            index=Index(('x', 'y'), name='foo'),
            name='f2',
        )

        f3 = sf.Frame.from_overlay((f1, f2))
        self.assertEqual(f3.index.name, 'foo')

        self.assertEqual(
            f3.to_pairs(),
            (
                ('a', (('x', 1.0), ('y', 200.0))),
                ('b', (('x', 4.0), ('y', 20.0))),
                ('c', (('x', False), ('y', False))),
            ),
        )

    def test_frame_from_overlay_d(self) -> None:
        f1 = Frame.from_dict(
            dict(a=(1, np.nan, 12), b=(np.nan, 20, 43), c=(False, None, False)),
            index=Index(('x', 'y', 'z'), name='foo'),
            name='f1',
        )
        f2 = Frame.from_dict(
            dict(a=(1, 200, 13), b=(4, 6, 23), c=(False, False, True)),
            index=Index(('x', 'y', 'z'), name='foo'),
            name='f2',
        )

        f3 = sf.Frame.from_overlay((f1, f2), index=('y', 'z'), columns=('b', 'c'))
        self.assertEqual(
            f3.to_pairs(),
            (('b', (('y', 20.0), ('z', 43.0))), ('c', (('y', False), ('z', False)))),
        )

    def test_frame_from_overlay_e(self) -> None:
        f = sf.Frame.from_overlay((ff.parse(f's({n},2)') for n in range(1, 4)))
        self.assertEqual(f.shape, (3, 2))
        self.assertTrue(f.equals(ff.parse('s(3,2)')))

    def test_frame_from_overlay_f(self) -> None:
        f1 = FrameGO.from_element('a', index=[1, 2, 3, 4], columns=list('abcd'))
        f2 = FrameGO.from_element('b', index=[1, 2, 3, 4], columns=list('abcd'))

        f3 = Frame.from_overlay((f1, f2))
        self.assertEqual(f3.shape, (4, 4))
        self.assertIs(f3.__class__, Frame)

    def test_frame_from_overlay_g(self) -> None:
        f1 = Frame.from_dict(dict(a=(1, 5), b=(5, 20)))
        f2 = Frame.from_dict(dict(a=(7, 200), b=(100, 6)))

        f3 = sf.Frame.from_overlay((f1, f2), func=lambda a: a == 5)
        self.assertEqual(
            f3.to_pairs(), (('a', ((0, 1), (1, 200))), ('b', ((0, 100), (1, 20))))
        )

    def test_frame_from_overlay_h(self) -> None:
        f1 = Frame.from_dict(dict(a=(1, -1), b=(-1, 20)))
        f2 = Frame.from_dict(dict(a=(7, 200), b=(100, 6), c=(-1, -1)))
        f3 = Frame.from_dict(dict(a=(7, 200), b=(100, 6), c=(30, 40)))

        f4 = sf.Frame.from_overlay((f1, f2, f3), func=lambda a: a < 0, fill_value=-1)
        self.assertEqual(
            f4.to_pairs(),
            (
                ('a', ((0, 1), (1, 200))),
                ('b', ((0, 100), (1, 20))),
                ('c', ((0, 30), (1, 40))),
            ),
        )

    def test_frame_from_overlay_i(self) -> None:
        f1 = Frame.from_dict(
            dict(
                a=(10, 20),
            )
        )
        f2 = Frame.from_dict(
            dict(
                b=(True, True, True),
            )
        )
        f3 = Frame.from_dict(
            dict(
                c=('a', 'b'),
            )
        )

        with WarningsSilent():
            f4 = sf.Frame.from_overlay(
                (f1, f2, f3),
                func=(lambda a: (a == 0) | (a == False) | (a == '')),
                fill_value=dict(a=0, b=False, c=''),
            )

        self.assertEqual(
            f4.to_pairs(),
            (
                ('a', ((0, 10), (1, 20), (2, 0))),
                ('b', ((0, True), (1, True), (2, True))),
                ('c', ((0, 'a'), (1, 'b'), (2, ''))),
            ),
        )

    # ---------------------------------------------------------------------------

    def test_frame_from_fields_a(self) -> None:
        f1 = sf.Frame.from_fields(([3, 4], [3, 'foo'], ['foo', 'bar']))
        self.assertEqual(
            f1.to_pairs(),
            (
                (0, ((0, 3), (1, 4))),
                (1, ((0, 3), (1, 'foo'))),
                (2, ((0, 'foo'), (1, 'bar'))),
            ),
        )

        with self.assertRaises(ErrorInitFrame):
            _ = sf.Frame.from_fields(
                ([3, 4], [3, 'foo'], ['foo', 'bar']), columns=('a', 'b')
            )

        f2 = sf.Frame.from_fields(
            ([3, 4], [3, 'foo'], ['foo', 'bar']), columns=('a', 'b', 'c')
        )
        self.assertEqual(
            f2.to_pairs(),
            (
                ('a', ((0, 3), (1, 4))),
                ('b', ((0, 3), (1, 'foo'))),
                ('c', ((0, 'foo'), (1, 'bar'))),
            ),
        )

        with self.assertRaises(ErrorInitFrame):
            _ = sf.Frame.from_fields(
                ([3, 4], [3, 'foo'], ['foo', 'bar']),
                columns=('a', 'b', 'c'),
                index=('x',),
            )

        f3 = sf.Frame.from_fields(
            ([3, 4], [3, 'foo'], ['foo', 'bar']),
            columns=('a', 'b', 'c'),
            index=('x', 'y'),
        )
        self.assertEqual(
            f3.to_pairs(),
            (
                ('a', (('x', 3), ('y', 4))),
                ('b', (('x', 3), ('y', 'foo'))),
                ('c', (('x', 'foo'), ('y', 'bar'))),
            ),
        )

        f4 = sf.Frame.from_fields(
            ([3, 4], [3, 'foo'], ['foo', 'bar']),
            columns=('a', 'b', 'c'),
            index=('x', 'y'),
            dtypes=str,
        )
        self.assertEqual(
            f4.to_pairs(),
            (
                ('a', (('x', '3'), ('y', '4'))),
                ('b', (('x', '3'), ('y', 'foo'))),
                ('c', (('x', 'foo'), ('y', 'bar'))),
            ),
        )

    def test_frame_from_fields_b(self) -> None:
        # test providing Series
        s1 = Series((3, 4, 5), index=('a', 'b', 'c'))
        s2 = Series((33, 54), index=('a', 'b'))
        s3 = Series((400, 300), index=('b', 'a'))

        with self.assertRaises(ErrorInitFrame):
            Frame.from_fields((s1, s2, s3))

        f1 = Frame.from_fields((s1, s2, s3), index=('b', 'a'))

        self.assertEqual(
            f1.to_pairs(),
            (
                (0, (('b', 4), ('a', 3))),
                (1, (('b', 54), ('a', 33))),
                (2, (('b', 400), ('a', 300))),
            ),
        )

        f2 = Frame.from_fields((s1, s2, s3), index=('b', 'a'), dtypes=(str, bool, str))
        self.assertEqual(
            f2.to_pairs(),
            (
                (0, (('b', '4'), ('a', '3'))),
                (1, (('b', True), ('a', True))),
                (2, (('b', '400'), ('a', '300'))),
            ),
        )

    def test_frame_from_fields_c(self) -> None:
        # test providing Series
        s1 = (3, 5)
        s2 = (33, 54)
        s3 = (400, 300)

        f = Frame.from_fields((s1, s2, s3), dtypes=str)
        self.assertEqual(
            f.to_pairs(),
            (
                (0, ((0, '3'), (1, '5'))),
                (1, ((0, '33'), (1, '54'))),
                (2, ((0, '400'), (1, '300'))),
            ),
        )

    def test_frame_from_fields_d(self) -> None:
        with self.assertRaises(ErrorInitFrame):
            f = Frame.from_fields((ff.parse('s(2,2)'), ff.parse('s(2,4)')))

    def test_frame_from_fields_e(self) -> None:
        f1 = sf.Frame.from_fields((np.array([3, 4]), np.array([3, 7])), dtypes=float)
        self.assertEqual(
            f1.dtypes.values.tolist(), [np.dtype('float64'), np.dtype('float64')]
        )
        self.assertEqual(
            f1.to_pairs(), ((0, ((0, 3.0), (1, 4.0))), (1, ((0, 3.0), (1, 7.0))))
        )

    def test_frame_from_fields_f(self) -> None:
        f1 = sf.Frame.from_fields(
            (Series([3, 4]), Series([3, 7])),
            fill_value=[-1, -2],
            index=(2, 1, 0),
        )
        self.assertEqual(
            f1.to_pairs(),
            ((0, ((2, -1), (1, 4), (0, 3))), (1, ((2, -2), (1, 7), (0, 3)))),
        )

    def test_frame_from_fields_g(self) -> None:
        f1 = sf.Frame.from_fields(
            (Series([3, 4]), Series([True, False])),
            fill_value=FillValueAuto,
            index=(2, 1, 0),
        )
        self.assertEqual(
            f1.to_pairs(),
            ((0, ((2, 0), (1, 4), (0, 3))), (1, ((2, False), (1, False), (0, True)))),
        )

    # ---------------------------------------------------------------------------
    def test_frame_from_zero_size_shape_a(self) -> None:
        f1 = Frame._from_zero_size_shape(
            index=(),
            columns=('a', 'b', 'c'),
            dtypes=bool,
        )
        self.assertEqual(f1.dtypes.unique().tolist(), [np.dtype(bool)])
        self.assertEqual(f1.shape, (0, 3))

    def test_frame_from_zero_size_shape_b(self) -> None:
        f1 = Frame._from_zero_size_shape(
            index=(),
            columns=Index(('a', 'b', 'c')),
            dtypes=bool,
            own_columns=True,
        )
        self.assertEqual(f1.dtypes.unique().tolist(), [np.dtype(bool)])
        self.assertEqual(f1.shape, (0, 3))

    def test_frame_from_zero_size_shape_c(self) -> None:
        f1 = Frame._from_zero_size_shape(
            columns=(),
            index=Index(('a', 'b', 'c')),
            dtypes=bool,
            own_index=True,
        )
        # while we have a type, we have not columns, and thus cannot get back an type
        self.assertEqual(f1.dtypes.unique().tolist(), [])
        self.assertEqual(f1.shape, (3, 0))

    def test_frame_from_zero_size_shape_d(self) -> None:
        with self.assertRaises(RuntimeError):
            _ = Frame._from_zero_size_shape(
                columns=(1, 2),
                index=Index(('a', 'b', 'c')),
                dtypes=bool,
            )

    def test_frame_from_zero_size_shape_e(self) -> None:
        f1 = Frame._from_zero_size_shape(
            index=(),
            columns=Index(('a', 'b', 'c')),
            dtypes=(bool, bool, str),
            own_columns=True,
        )
        self.assertEqual(
            f1.dtypes.values.tolist(), [np.dtype(bool), np.dtype(bool), np.dtype('U1')]
        )
        self.assertEqual(f1.shape, (0, 3))

    def test_frame_from_zero_size_shape_f(self) -> None:
        f1 = Frame._from_zero_size_shape(
            index=(),
            columns=Index(('a', 'b', 'c')),
            dtypes=(bool, bool, str),
            own_columns=True,
        )
        self.assertEqual(
            f1.dtypes.values.tolist(), [np.dtype(bool), np.dtype(bool), np.dtype('U1')]
        )
        self.assertEqual(f1.shape, (0, 3))

    # ---------------------------------------------------------------------------

    def test_frame_sample_a(self) -> None:
        f = ff.parse('s(20,10)|i(I,str)|c(I,str)')

        self.assertEqual(
            f.sample(4, 2, seed=2).to_pairs(),
            (
                (
                    'ztsv',
                    (
                        ('zZbu', -610.8),
                        ('zmVj', -3367.74),
                        ('zGDJ', 1146.32),
                        ('zB7E', 1625.5),
                    ),
                ),
                (
                    'zmVj',
                    (
                        ('zZbu', 3511.58),
                        ('zmVj', 647.9),
                        ('zGDJ', 3367.24),
                        ('zB7E', 1459.94),
                    ),
                ),
            ),
        )

        self.assertEqual(
            f.sample(columns=1, seed=2).to_pairs(),
            (
                (
                    'zmVj',
                    (
                        ('zZbu', 3511.58),
                        ('ztsv', 1175.36),
                        ('zUvW', 2925.68),
                        ('zkuW', 3408.8),
                        ('zmVj', 647.9),
                        ('z2Oo', 2755.18),
                        ('z5l6', -1259.28),
                        ('zCE3', 3442.84),
                        ('zr4u', -3093.72),
                        ('zYVB', 2520.5),
                        ('zOyq', 1194.56),
                        ('zIA5', -1957.02),
                        ('zGDJ', 3367.24),
                        ('zmhG', 2600.2),
                        ('zo2Q', -3011.46),
                        ('zjZQ', -3148.74),
                        ('zO5l', 713.68),
                        ('zEdH', -555.42),
                        ('zB7E', 1459.94),
                        ('zwIp', 3287.02),
                    ),
                ),
            ),
        )

    def test_frame_sample_b(self) -> None:
        f = ff.parse('s(20,10)|i(IH,(str,int))|c(IH,(str,int))')

        self.assertEqual(
            f.sample(2, 3, seed=3).to_pairs(),
            (
                (
                    ('zZbu', 119909),
                    ((('ztsv', 194224), -823.14), (('zCE3', 137759), 3822.16)),
                ),
                (
                    ('zUvW', 96520),
                    ((('ztsv', 194224), 2925.68), (('zCE3', 137759), -3011.46)),
                ),
                (
                    ('zUvW', -88017),
                    ((('ztsv', 194224), 268.96), (('zCE3', 137759), -1957.02)),
                ),
            ),
        )

    def test_frame_sample_c(self) -> None:
        f = ff.parse('s(20,10)|i(IH,(str,int))|c(IH,(str,int))')
        post = f.sample(seed=3)
        self.assertEqual(post.shape, (20, 10))

    # ---------------------------------------------------------------------------

    def test_frame_via_T_add_a(self) -> None:
        f1 = ff.parse('s(6,3)|v(int)')

        f2 = f1.via_T + (10, 10, 0, 10, 10, 0)
        self.assertEqual(
            f2.to_pairs(),
            (
                (
                    0,
                    (
                        (0, -88007),
                        (1, 92877),
                        (2, 84967),
                        (3, 13458),
                        (4, 175589),
                        (5, 58768),
                    ),
                ),
                (
                    1,
                    (
                        (0, 162207),
                        (1, -41147),
                        (2, 5729),
                        (3, -168377),
                        (4, 140637),
                        (5, 66269),
                    ),
                ),
                (
                    2,
                    (
                        (0, -3638),
                        (1, 91311),
                        (2, 30205),
                        (3, 54030),
                        (4, 129027),
                        (5, 35021),
                    ),
                ),
            ),
        )

        f3 = (10, 10, 0, 10, 10, 0) + f1.via_T
        self.assertEqual(
            f3.to_pairs(),
            (
                (
                    0,
                    (
                        (0, -88007),
                        (1, 92877),
                        (2, 84967),
                        (3, 13458),
                        (4, 175589),
                        (5, 58768),
                    ),
                ),
                (
                    1,
                    (
                        (0, 162207),
                        (1, -41147),
                        (2, 5729),
                        (3, -168377),
                        (4, 140637),
                        (5, 66269),
                    ),
                ),
                (
                    2,
                    (
                        (0, -3638),
                        (1, 91311),
                        (2, 30205),
                        (3, 54030),
                        (4, 129027),
                        (5, 35021),
                    ),
                ),
            ),
        )

    def test_frame_via_T_sub_a(self) -> None:
        f1 = ff.parse('s(6,3)|v(int)')

        f2 = f1.via_T - (10, 10, 0, 10, 10, 0)

        self.assertEqual(
            f2.to_pairs(),
            (
                (
                    0,
                    (
                        (0, -88027),
                        (1, 92857),
                        (2, 84967),
                        (3, 13438),
                        (4, 175569),
                        (5, 58768),
                    ),
                ),
                (
                    1,
                    (
                        (0, 162187),
                        (1, -41167),
                        (2, 5729),
                        (3, -168397),
                        (4, 140617),
                        (5, 66269),
                    ),
                ),
                (
                    2,
                    (
                        (0, -3658),
                        (1, 91291),
                        (2, 30205),
                        (3, 54010),
                        (4, 129007),
                        (5, 35021),
                    ),
                ),
            ),
        )

        f3 = (10, 10, 0, 10, 10, 0) - f1.via_T
        self.assertEqual(
            f3.to_pairs(),
            (
                (
                    0,
                    (
                        (0, 88027),
                        (1, -92857),
                        (2, -84967),
                        (3, -13438),
                        (4, -175569),
                        (5, -58768),
                    ),
                ),
                (
                    1,
                    (
                        (0, -162187),
                        (1, 41167),
                        (2, -5729),
                        (3, 168397),
                        (4, -140617),
                        (5, -66269),
                    ),
                ),
                (
                    2,
                    (
                        (0, 3658),
                        (1, -91291),
                        (2, -30205),
                        (3, -54010),
                        (4, -129007),
                        (5, -35021),
                    ),
                ),
            ),
        )

    def test_frame_via_T_mul_a(self) -> None:
        f1 = ff.parse('s(6,3)|v(int)')

        f2 = f1.via_T * (1, 0, 0, 1, 0, 0)
        self.assertEqual(
            f2.to_pairs(),
            (
                (0, ((0, -88017), (1, 0), (2, 0), (3, 13448), (4, 0), (5, 0))),
                (1, ((0, 162197), (1, 0), (2, 0), (3, -168387), (4, 0), (5, 0))),
                (2, ((0, -3648), (1, 0), (2, 0), (3, 54020), (4, 0), (5, 0))),
            ),
        )

        f3 = (1, 0, 0, 1, 0, 0) * f1.via_T
        self.assertEqual(
            f3.to_pairs(),
            (
                (0, ((0, -88017), (1, 0), (2, 0), (3, 13448), (4, 0), (5, 0))),
                (1, ((0, 162197), (1, 0), (2, 0), (3, -168387), (4, 0), (5, 0))),
                (2, ((0, -3648), (1, 0), (2, 0), (3, 54020), (4, 0), (5, 0))),
            ),
        )

    def test_frame_via_T_truediv_a(self) -> None:
        f1 = ff.parse('s(6,3)|v(int)')

        f2 = f1.via_T / (1, 2, 2, 2, 2, 1)

        self.assertEqual(
            f2.to_pairs(),
            (
                (
                    0,
                    (
                        (0, -88017.0),
                        (1, 46433.5),
                        (2, 42483.5),
                        (3, 6724.0),
                        (4, 87789.5),
                        (5, 58768.0),
                    ),
                ),
                (
                    1,
                    (
                        (0, 162197.0),
                        (1, -20578.5),
                        (2, 2864.5),
                        (3, -84193.5),
                        (4, 70313.5),
                        (5, 66269.0),
                    ),
                ),
                (
                    2,
                    (
                        (0, -3648.0),
                        (1, 45650.5),
                        (2, 15102.5),
                        (3, 27010.0),
                        (4, 64508.5),
                        (5, 35021.0),
                    ),
                ),
            ),
        )

        f3 = (10000, 20000, 20000, 20000, 20000, 10000) / f1.via_T
        self.assertEqual(
            round(f3, 1).to_pairs(),
            (
                (0, ((0, -0.1), (1, 0.2), (2, 0.2), (3, 1.5), (4, 0.1), (5, 0.2))),
                (1, ((0, 0.1), (1, -0.5), (2, 3.5), (3, -0.1), (4, 0.1), (5, 0.2))),
                (2, ((0, -2.7), (1, 0.2), (2, 0.7), (3, 0.4), (4, 0.2), (5, 0.3))),
            ),
        )

    def test_frame_via_T_floordiv_a(self) -> None:
        f1 = ff.parse('s(6,3)|v(int)')

        f2 = f1.via_T // (2, 2, 2, 2, 2, 2)
        self.assertEqual(
            f2.to_pairs(),
            (
                (
                    0,
                    (
                        (0, -44009),
                        (1, 46433),
                        (2, 42483),
                        (3, 6724),
                        (4, 87789),
                        (5, 29384),
                    ),
                ),
                (
                    1,
                    (
                        (0, 81098),
                        (1, -20579),
                        (2, 2864),
                        (3, -84194),
                        (4, 70313),
                        (5, 33134),
                    ),
                ),
                (
                    2,
                    (
                        (0, -1824),
                        (1, 45650),
                        (2, 15102),
                        (3, 27010),
                        (4, 64508),
                        (5, 17510),
                    ),
                ),
            ),
        )

        f3 = (10000, 20000, 20000, 20000, 20000, 10000) // f1.via_T
        self.assertEqual(
            f3.to_pairs(),
            (
                (0, ((0, -1), (1, 0), (2, 0), (3, 1), (4, 0), (5, 0))),
                (1, ((0, 0), (1, -1), (2, 3), (3, -1), (4, 0), (5, 0))),
                (2, ((0, -3), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0))),
            ),
        )

    def test_frame_via_T_mod_a(self) -> None:
        f1 = ff.parse('s(6,3)|v(int)')

        f2 = f1.via_T % (2, 2, 3, 4, 8, 9)

        self.assertEqual(
            f2.to_pairs(),
            (
                (0, ((0, 1), (1, 1), (2, 1), (3, 0), (4, 3), (5, 7))),
                (1, ((0, 1), (1, 1), (2, 2), (3, 1), (4, 3), (5, 2))),
                (2, ((0, 0), (1, 1), (2, 1), (3, 0), (4, 1), (5, 2))),
            ),
        )

    def test_frame_via_T_pow_a(self) -> None:
        f1 = ff.parse('s(6,3)|v(int)')

        f2 = f1.via_T ** np.array((2, 2, 1, 1, 2, 2), dtype=np.int64)

        self.assertEqual(
            f2.to_pairs(),
            (
                (
                    0,
                    (
                        (0, 7746992289),
                        (1, 8624279689),
                        (2, 84967),
                        (3, 13448),
                        (4, 30827985241),
                        (5, 3453677824),
                    ),
                ),
                (
                    1,
                    (
                        (0, 26307866809),
                        (1, 1693898649),
                        (2, 5729),
                        (3, -168387),
                        (4, 19775953129),
                        (5, 4391580361),
                    ),
                ),
                (
                    2,
                    (
                        (0, 13307904),
                        (1, 8335872601),
                        (2, 30205),
                        (3, 54020),
                        (4, 16645386289),
                        (5, 1226470441),
                    ),
                ),
            ),
        )

    def test_frame_via_T_lshift_a(self) -> None:
        f1 = ff.parse('s(6,3)|v(int)')

        f2 = f1.via_T << (2, 2, 2, 2, 2, 2)

        self.assertEqual(
            f2.to_pairs(),
            (
                (
                    0,
                    (
                        (0, -352068),
                        (1, 371468),
                        (2, 339868),
                        (3, 53792),
                        (4, 702316),
                        (5, 235072),
                    ),
                ),
                (
                    1,
                    (
                        (0, 648788),
                        (1, -164628),
                        (2, 22916),
                        (3, -673548),
                        (4, 562508),
                        (5, 265076),
                    ),
                ),
                (
                    2,
                    (
                        (0, -14592),
                        (1, 365204),
                        (2, 120820),
                        (3, 216080),
                        (4, 516068),
                        (5, 140084),
                    ),
                ),
            ),
        )

    def test_frame_via_T_rshift_a(self) -> None:
        f1 = ff.parse('s(6,3)|v(int)')

        f2 = f1.via_T >> (2, 2, 2, 2, 2, 2)

        self.assertEqual(
            f2.to_pairs(),
            (
                (
                    0,
                    (
                        (0, -22005),
                        (1, 23216),
                        (2, 21241),
                        (3, 3362),
                        (4, 43894),
                        (5, 14692),
                    ),
                ),
                (
                    1,
                    (
                        (0, 40549),
                        (1, -10290),
                        (2, 1432),
                        (3, -42097),
                        (4, 35156),
                        (5, 16567),
                    ),
                ),
                (
                    2,
                    (
                        (0, -912),
                        (1, 22825),
                        (2, 7551),
                        (3, 13505),
                        (4, 32254),
                        (5, 8755),
                    ),
                ),
            ),
        )

    def test_frame_via_T_and_a(self) -> None:
        f1 = ff.parse('s(6,3)|v(int)')

        f2 = f1.via_T & (1, 0, 0, 1, 0, 0)
        self.assertEqual(
            f2.to_pairs(),
            (
                (0, ((0, 1), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0))),
                (1, ((0, 1), (1, 0), (2, 0), (3, 1), (4, 0), (5, 0))),
                (2, ((0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0))),
            ),
        )

    def test_frame_via_T_xor_a(self) -> None:
        f1 = ff.parse('s(6,3)|v(int)')

        f2 = (f1 > 0).via_T ^ (True, False, True, False, False, False)

        self.assertEqual(
            f2.to_pairs(),
            (
                (
                    0,
                    ((0, True), (1, True), (2, False), (3, True), (4, True), (5, True)),
                ),
                (
                    1,
                    (
                        (0, False),
                        (1, False),
                        (2, False),
                        (3, False),
                        (4, True),
                        (5, True),
                    ),
                ),
                (
                    2,
                    ((0, True), (1, True), (2, False), (3, True), (4, True), (5, True)),
                ),
            ),
        )

    def test_frame_via_T_or_a(self) -> None:
        f1 = ff.parse('s(6,3)|v(int)')

        f2 = (f1 < 0).via_T | (True, False, False, False, False, True)

        self.assertEqual(
            f2.to_pairs(),
            (
                (
                    0,
                    (
                        (0, True),
                        (1, False),
                        (2, False),
                        (3, False),
                        (4, False),
                        (5, True),
                    ),
                ),
                (
                    1,
                    (
                        (0, True),
                        (1, True),
                        (2, False),
                        (3, True),
                        (4, False),
                        (5, True),
                    ),
                ),
                (
                    2,
                    (
                        (0, True),
                        (1, False),
                        (2, False),
                        (3, False),
                        (4, False),
                        (5, True),
                    ),
                ),
            ),
        )

    def test_frame_via_T_eq_a(self) -> None:
        f1 = ff.parse('s(6,3)|v(int)')

        f2 = f1.via_T == f1[1]
        self.assertEqual(
            f2.to_pairs(),
            (
                (
                    0,
                    (
                        (0, False),
                        (1, False),
                        (2, False),
                        (3, False),
                        (4, False),
                        (5, False),
                    ),
                ),
                (1, ((0, True), (1, True), (2, True), (3, True), (4, True), (5, True))),
                (
                    2,
                    (
                        (0, False),
                        (1, False),
                        (2, False),
                        (3, False),
                        (4, False),
                        (5, False),
                    ),
                ),
            ),
        )

    def test_frame_via_T_ne_a(self) -> None:
        f1 = ff.parse('s(6,3)|v(int)')

        f2 = f1.via_T != f1[1]
        self.assertEqual(
            f2.to_pairs(),
            (
                (0, ((0, True), (1, True), (2, True), (3, True), (4, True), (5, True))),
                (
                    1,
                    (
                        (0, False),
                        (1, False),
                        (2, False),
                        (3, False),
                        (4, False),
                        (5, False),
                    ),
                ),
                (2, ((0, True), (1, True), (2, True), (3, True), (4, True), (5, True))),
            ),
        )

    def test_frame_via_T_lt_a(self) -> None:
        f1 = ff.parse('s(6,3)|v(int)')

        f2 = f1.via_T < f1[1]

        self.assertEqual(
            f2.to_pairs(),
            (
                (
                    0,
                    (
                        (0, True),
                        (1, False),
                        (2, False),
                        (3, False),
                        (4, False),
                        (5, True),
                    ),
                ),
                (
                    1,
                    (
                        (0, False),
                        (1, False),
                        (2, False),
                        (3, False),
                        (4, False),
                        (5, False),
                    ),
                ),
                (
                    2,
                    (
                        (0, True),
                        (1, False),
                        (2, False),
                        (3, False),
                        (4, True),
                        (5, True),
                    ),
                ),
            ),
        )

    def test_frame_via_T_le_a(self) -> None:
        f1 = ff.parse('s(6,3)|v(int)')

        f2 = f1.via_T <= f1[1]

        self.assertEqual(
            f2.to_pairs(),
            (
                (
                    0,
                    (
                        (0, True),
                        (1, False),
                        (2, False),
                        (3, False),
                        (4, False),
                        (5, True),
                    ),
                ),
                (1, ((0, True), (1, True), (2, True), (3, True), (4, True), (5, True))),
                (
                    2,
                    (
                        (0, True),
                        (1, False),
                        (2, False),
                        (3, False),
                        (4, True),
                        (5, True),
                    ),
                ),
            ),
        )

    def test_frame_via_T_gt_a(self) -> None:
        f1 = ff.parse('s(6,3)|v(int)')

        f2 = f1.via_T > f1[1]

        self.assertEqual(
            f2.to_pairs(),
            (
                (
                    0,
                    (
                        (0, False),
                        (1, True),
                        (2, True),
                        (3, True),
                        (4, True),
                        (5, False),
                    ),
                ),
                (
                    1,
                    (
                        (0, False),
                        (1, False),
                        (2, False),
                        (3, False),
                        (4, False),
                        (5, False),
                    ),
                ),
                (
                    2,
                    (
                        (0, False),
                        (1, True),
                        (2, True),
                        (3, True),
                        (4, False),
                        (5, False),
                    ),
                ),
            ),
        )

    def test_frame_via_T_ge_a(self) -> None:
        f1 = ff.parse('s(6,3)|v(int)')

        f2 = f1.via_T >= f1[1]

        self.assertEqual(
            f2.to_pairs(),
            (
                (
                    0,
                    (
                        (0, False),
                        (1, True),
                        (2, True),
                        (3, True),
                        (4, True),
                        (5, False),
                    ),
                ),
                (1, ((0, True), (1, True), (2, True), (3, True), (4, True), (5, True))),
                (
                    2,
                    (
                        (0, False),
                        (1, True),
                        (2, True),
                        (3, True),
                        (4, False),
                        (5, False),
                    ),
                ),
            ),
        )

    # ---------------------------------------------------------------------------

    def test_frame_deepcopy_a(self) -> None:
        f1 = sf.FrameGO(index=tuple('abc'))
        f1['x'] = (3, 4, 5)
        f1['y'] = Series.from_dict(dict(b=10, c=11, a=12))

        f2 = copy.deepcopy(f1)
        f2['z'] = 0
        f1['q'] = 10

        self.assertEqual(
            f2.to_pairs(),
            (
                ('x', (('a', 3), ('b', 4), ('c', 5))),
                ('y', (('a', 12), ('b', 10), ('c', 11))),
                ('z', (('a', 0), ('b', 0), ('c', 0))),
            ),
        )

        self.assertTrue(
            [id(b) for b in f1._blocks._blocks] != [id(b) for b in f2._blocks._blocks]
        )

    def test_frame_deepcopy_b(self) -> None:
        a1 = np.array(list('abc'))
        a1.flags.writeable = False
        f1 = Frame.from_fields((a1, a1, a1), index=a1, columns=a1)

        a1_id = id(a1)
        self.assertEqual(id(f1.index.values), a1_id)
        self.assertEqual(id(f1.columns.values), a1_id)
        self.assertEqual(id(f1._blocks._blocks[0]), a1_id)
        self.assertEqual(id(f1._blocks._blocks[1]), a1_id)
        self.assertEqual(id(f1._blocks._blocks[2]), a1_id)

        f2 = copy.deepcopy(f1)
        a2_id = id(f2.index.values)

        self.assertEqual(id(f2.columns.values), a2_id)
        self.assertEqual(id(f2._blocks._blocks[0]), a2_id)
        self.assertEqual(id(f2._blocks._blocks[1]), a2_id)
        self.assertEqual(id(f2._blocks._blocks[2]), a2_id)

    # ---------------------------------------------------------------------------

    def test_frame_relabel_shift_in_a(self) -> None:
        f1 = ff.parse('s(5,4)|i(I,int)|c(I,str)|v(str)')

        f2 = f1.relabel_shift_in('zUvW', axis=0)
        self.assertEqual(
            f2.to_pairs(),
            (
                (
                    'zZbu',
                    (
                        ((34715, 'ztsv'), 'zjZQ'),
                        ((-3648, 'zUvW'), 'zO5l'),
                        ((91301, 'zkuW'), 'zEdH'),
                        ((30205, 'zmVj'), 'zB7E'),
                        ((54020, 'z2Oo'), 'zwIp'),
                    ),
                ),
                (
                    'ztsv',
                    (
                        ((34715, 'ztsv'), 'zaji'),
                        ((-3648, 'zUvW'), 'zJnC'),
                        ((91301, 'zkuW'), 'zDdR'),
                        ((30205, 'zmVj'), 'zuVU'),
                        ((54020, 'z2Oo'), 'zKka'),
                    ),
                ),
                (
                    'zkuW',
                    (
                        ((34715, 'ztsv'), 'z2Oo'),
                        ((-3648, 'zUvW'), 'z5l6'),
                        ((91301, 'zkuW'), 'zCE3'),
                        ((30205, 'zmVj'), 'zr4u'),
                        ((54020, 'z2Oo'), 'zYVB'),
                    ),
                ),
            ),
        )
        self.assertEqual(f2.index.name, ('__index0__', 'zUvW'))

        f3 = f1.relabel_shift_in(30205, axis=1)
        self.assertEqual(
            f3.to_pairs(),
            (
                (
                    ('zZbu', 'zB7E'),
                    (
                        (34715, 'zjZQ'),
                        (-3648, 'zO5l'),
                        (91301, 'zEdH'),
                        (54020, 'zwIp'),
                    ),
                ),
                (
                    ('ztsv', 'zuVU'),
                    (
                        (34715, 'zaji'),
                        (-3648, 'zJnC'),
                        (91301, 'zDdR'),
                        (54020, 'zKka'),
                    ),
                ),
                (
                    ('zUvW', 'zmVj'),
                    (
                        (34715, 'ztsv'),
                        (-3648, 'zUvW'),
                        (91301, 'zkuW'),
                        (54020, 'z2Oo'),
                    ),
                ),
                (
                    ('zkuW', 'zr4u'),
                    (
                        (34715, 'z2Oo'),
                        (-3648, 'z5l6'),
                        (91301, 'zCE3'),
                        (54020, 'zYVB'),
                    ),
                ),
            ),
        )
        self.assertEqual(f3.columns.name, ('__index0__', 30205))

    def test_frame_relabel_shift_in_b(self) -> None:
        f1 = ff.parse('s(5,4)|i(I,int)|c(I,str)|v(str)')

        f2 = f1.relabel_shift_in(['ztsv', 'zkuW'], axis=0)
        self.assertEqual(
            f2.to_pairs(),
            (
                (
                    'zZbu',
                    (
                        ((34715, 'zaji', 'z2Oo'), 'zjZQ'),
                        ((-3648, 'zJnC', 'z5l6'), 'zO5l'),
                        ((91301, 'zDdR', 'zCE3'), 'zEdH'),
                        ((30205, 'zuVU', 'zr4u'), 'zB7E'),
                        ((54020, 'zKka', 'zYVB'), 'zwIp'),
                    ),
                ),
                (
                    'zUvW',
                    (
                        ((34715, 'zaji', 'z2Oo'), 'ztsv'),
                        ((-3648, 'zJnC', 'z5l6'), 'zUvW'),
                        ((91301, 'zDdR', 'zCE3'), 'zkuW'),
                        ((30205, 'zuVU', 'zr4u'), 'zmVj'),
                        ((54020, 'zKka', 'zYVB'), 'z2Oo'),
                    ),
                ),
            ),
        )
        self.assertEqual(f2.index.name, ('__index0__', 'ztsv', 'zkuW'))

        f3 = f1.relabel_shift_in([34715, 54020], axis=1)
        self.assertEqual(
            f3.to_pairs(),
            (
                (
                    ('zZbu', 'zjZQ', 'zwIp'),
                    ((-3648, 'zO5l'), (91301, 'zEdH'), (30205, 'zB7E')),
                ),
                (
                    ('ztsv', 'zaji', 'zKka'),
                    ((-3648, 'zJnC'), (91301, 'zDdR'), (30205, 'zuVU')),
                ),
                (
                    ('zUvW', 'ztsv', 'z2Oo'),
                    ((-3648, 'zUvW'), (91301, 'zkuW'), (30205, 'zmVj')),
                ),
                (
                    ('zkuW', 'z2Oo', 'zYVB'),
                    ((-3648, 'z5l6'), (91301, 'zCE3'), (30205, 'zr4u')),
                ),
            ),
        )

        self.assertEqual(f3.columns.name, ('__index0__', 34715, 54020))

    def test_frame_relabel_shift_in_c(self) -> None:
        f1 = ff.parse('s(5,4)|i(I,int)|c(I,str)|v(str)')
        f2 = f1.relabel_shift_in(slice(None), axis=0)
        self.assertEqual(f2.shape, (5, 0))
        self.assertEqual(f2.index.name, ('__index0__', 'zZbu', 'ztsv', 'zUvW', 'zkuW'))

        f3 = f1.relabel_shift_in(slice(None), axis=1)
        self.assertEqual(f3.shape, (0, 4))
        self.assertEqual(
            f3.columns.name, ('__index0__', 34715, -3648, 91301, 30205, 54020)
        )

    def test_frame_relabel_shift_in_d(self) -> None:
        f1 = ff.parse('s(3,4)|i(IH,(int,str))|c(IH,(str,int))|v(str)')

        f2 = f1.relabel_shift_in([('zZbu', 119909), ('ztsv', 172133)])
        self.assertEqual(f1.shape, (3, 4))
        self.assertEqual(
            f2.to_pairs(),
            (
                (
                    ('zZbu', 105269),
                    (
                        ((34715, 'zOyq', 'zaji', 'z2Oo'), 'zjZQ'),
                        ((34715, 'zIA5', 'zJnC', 'z5l6'), 'zO5l'),
                        ((-3648, 'zGDJ', 'zDdR', 'zCE3'), 'zEdH'),
                    ),
                ),
                (
                    ('ztsv', 194224),
                    (
                        ((34715, 'zOyq', 'zaji', 'z2Oo'), 'ztsv'),
                        ((34715, 'zIA5', 'zJnC', 'z5l6'), 'zUvW'),
                        ((-3648, 'zGDJ', 'zDdR', 'zCE3'), 'zkuW'),
                    ),
                ),
            ),
        )
        self.assertEqual(
            f2.index.name,
            ('__index0__', '__index1__', ('zZbu', 119909), ('ztsv', 172133)),
        )

        f3 = f1.relabel_shift_in(slice((34715, 'zOyq'), None), axis=1)
        self.assertEqual(f1.shape, (3, 4))
        self.assertEqual(
            f3.to_pairs(),
            (
                (('zZbu', 105269, 'zjZQ', 'zO5l', 'zEdH'), ()),
                (('zZbu', 119909, 'zaji', 'zJnC', 'zDdR'), ()),
                (('ztsv', 194224, 'ztsv', 'zUvW', 'zkuW'), ()),
                (('ztsv', 172133, 'z2Oo', 'z5l6', 'zCE3'), ()),
            ),
        )
        self.assertEqual(
            f3.columns.name,
            (
                '__index0__',
                '__index1__',
                (34715, 'zOyq'),
                (34715, 'zIA5'),
                (-3648, 'zGDJ'),
            ),
        )

    def test_frame_relabel_shift_in_e(self) -> None:
        f1 = ff.parse('f(Fg)|s(3,4)|i(I,int)|c(IHg,(str,int))|v(str)').rename(
            index='a', columns=('x', 'y')
        )

        f2 = f1.relabel_shift_in(('zZbu', 119909), axis=0)
        self.assertEqual(f2.index.name, ('a', ('zZbu', 119909)))
        self.assertTrue(f2.__class__, FrameGO)

        f3 = f1.relabel_shift_in(-3648, axis=1)
        self.assertEqual(f3.columns.name, ('x', 'y', -3648))

        self.assertEqual(
            f3.to_pairs(),
            (
                (('zZbu', 105269, 'zO5l'), ((34715, 'zjZQ'), (91301, 'zEdH'))),
                (('zZbu', 119909, 'zJnC'), ((34715, 'zaji'), (91301, 'zDdR'))),
                (('ztsv', 194224, 'zUvW'), ((34715, 'ztsv'), (91301, 'zkuW'))),
                (('ztsv', 172133, 'z5l6'), ((34715, 'z2Oo'), (91301, 'zCE3'))),
            ),
        )

    def test_frame_relabel_shift_in_f(self) -> None:
        f1 = ff.parse('f(Fg)|s(3,4)|i(I,int)|c(IHg,(str,int))|v(str)').rename(
            index='a', columns=('x', 'y')
        )

        f1[('AAAA', 999999)] = 'abcd'

        f2 = f1.relabel_shift_in(('zZbu', 119909), axis=0)
        self.assertEqual(f2.index.name, ('a', ('zZbu', 119909)))
        self.assertTrue(f2.__class__, FrameGO)

        f3 = f1.relabel_shift_in(-3648, axis=1)
        self.assertEqual(f3.columns.name, ('x', 'y', -3648))

        self.assertEqual(
            f3.to_pairs(),
            (
                (('zZbu', 105269, 'zO5l'), ((34715, 'zjZQ'), (91301, 'zEdH'))),
                (('zZbu', 119909, 'zJnC'), ((34715, 'zaji'), (91301, 'zDdR'))),
                (('ztsv', 194224, 'zUvW'), ((34715, 'ztsv'), (91301, 'zkuW'))),
                (('ztsv', 172133, 'z5l6'), ((34715, 'z2Oo'), (91301, 'zCE3'))),
                (('AAAA', 999999, 'abcd'), ((34715, 'abcd'), (91301, 'abcd'))),
            ),
        )

    def test_frame_relabel_shift_in_g(self) -> None:
        f1 = ff.parse('s(5,4)|i(I,int)|c(I,str)|v(dtD, dtY)')
        f2 = f1.relabel_shift_in(sf.ILoc[:2], axis=0, index_constructors=IndexDate)

        self.assertEqual([dt.kind for dt in f2.index.dtypes.values], ['i', 'M', 'M'])

        f3 = f1.relabel_shift_in(
            sf.ILoc[:2], axis=0, index_constructors=(IndexDate, IndexYear)
        )
        self.assertEqual([dt.kind for dt in f3.index.dtypes.values], ['i', 'M', 'M'])

        with self.assertRaises(RuntimeError):
            _ = f1.relabel_shift_in(
                sf.ILoc[:2],
                axis=0,
                index_constructors=(IndexDate, IndexYear, IndexDate),
            )

    # ---------------------------------------------------------------------------

    def test_frame_relabel_shift_out_a(self) -> None:
        f1 = ff.parse('s(3,4)|i(I,int)|c(I,str)|v(str)')
        f2 = f1.relabel_shift_out(0, axis=0)

        self.assertEqual(
            f2.to_pairs(),
            (
                ('__index0__', ((0, 34715), (1, -3648), (2, 91301))),
                ('zZbu', ((0, 'zjZQ'), (1, 'zO5l'), (2, 'zEdH'))),
                ('ztsv', ((0, 'zaji'), (1, 'zJnC'), (2, 'zDdR'))),
                ('zUvW', ((0, 'ztsv'), (1, 'zUvW'), (2, 'zkuW'))),
                ('zkuW', ((0, 'z2Oo'), (1, 'z5l6'), (2, 'zCE3'))),
            ),
        )

        f3 = f1.relabel_shift_out(0, axis=1)
        self.assertEqual(
            f3.to_pairs(),
            (
                (
                    0,
                    (
                        ('__index0__', 'zZbu'),
                        (34715, 'zjZQ'),
                        (-3648, 'zO5l'),
                        (91301, 'zEdH'),
                    ),
                ),
                (
                    1,
                    (
                        ('__index0__', 'ztsv'),
                        (34715, 'zaji'),
                        (-3648, 'zJnC'),
                        (91301, 'zDdR'),
                    ),
                ),
                (
                    2,
                    (
                        ('__index0__', 'zUvW'),
                        (34715, 'ztsv'),
                        (-3648, 'zUvW'),
                        (91301, 'zkuW'),
                    ),
                ),
                (
                    3,
                    (
                        ('__index0__', 'zkuW'),
                        (34715, 'z2Oo'),
                        (-3648, 'z5l6'),
                        (91301, 'zCE3'),
                    ),
                ),
            ),
        )

    def test_frame_relabel_shift_out_b(self) -> None:
        f1 = ff.parse('s(3,4)|i(IH,(int,str))|c(IH,(str,int))|v(str)').rename(
            index=('a', 'b'), columns=('x', 'y')
        )

        f2 = f1.relabel_shift_out([0, 1], axis=0)
        self.assertEqual(
            f2.to_pairs(),
            (
                ('a', ((0, 34715), (1, 34715), (2, -3648))),
                ('b', ((0, 'zOyq'), (1, 'zIA5'), (2, 'zGDJ'))),
                (('zZbu', 105269), ((0, 'zjZQ'), (1, 'zO5l'), (2, 'zEdH'))),
                (('zZbu', 119909), ((0, 'zaji'), (1, 'zJnC'), (2, 'zDdR'))),
                (('ztsv', 194224), ((0, 'ztsv'), (1, 'zUvW'), (2, 'zkuW'))),
                (('ztsv', 172133), ((0, 'z2Oo'), (1, 'z5l6'), (2, 'zCE3'))),
            ),
        )
        self.assertEqual(f2.columns.name, ('x', 'y'))

        f3 = f1.relabel_shift_out(0, axis=0)
        self.assertEqual(
            f3.to_pairs(),
            (
                ('a', (('zOyq', 34715), ('zIA5', 34715), ('zGDJ', -3648))),
                (
                    ('zZbu', 105269),
                    (('zOyq', 'zjZQ'), ('zIA5', 'zO5l'), ('zGDJ', 'zEdH')),
                ),
                (
                    ('zZbu', 119909),
                    (('zOyq', 'zaji'), ('zIA5', 'zJnC'), ('zGDJ', 'zDdR')),
                ),
                (
                    ('ztsv', 194224),
                    (('zOyq', 'ztsv'), ('zIA5', 'zUvW'), ('zGDJ', 'zkuW')),
                ),
                (
                    ('ztsv', 172133),
                    (('zOyq', 'z2Oo'), ('zIA5', 'z5l6'), ('zGDJ', 'zCE3')),
                ),
            ),
        )
        self.assertEqual(f3.index.name, 'b')
        self.assertEqual(f3.columns.name, ('x', 'y'))

        f4 = f1.relabel_shift_out(0, axis=1)
        self.assertEqual(f4.index.name, ('a', 'b'))
        self.assertEqual(
            f4.to_pairs(),
            (
                (
                    105269,
                    (
                        ('x', 'zZbu'),
                        ((34715, 'zOyq'), 'zjZQ'),
                        ((34715, 'zIA5'), 'zO5l'),
                        ((-3648, 'zGDJ'), 'zEdH'),
                    ),
                ),
                (
                    119909,
                    (
                        ('x', 'zZbu'),
                        ((34715, 'zOyq'), 'zaji'),
                        ((34715, 'zIA5'), 'zJnC'),
                        ((-3648, 'zGDJ'), 'zDdR'),
                    ),
                ),
                (
                    194224,
                    (
                        ('x', 'ztsv'),
                        ((34715, 'zOyq'), 'ztsv'),
                        ((34715, 'zIA5'), 'zUvW'),
                        ((-3648, 'zGDJ'), 'zkuW'),
                    ),
                ),
                (
                    172133,
                    (
                        ('x', 'ztsv'),
                        ((34715, 'zOyq'), 'z2Oo'),
                        ((34715, 'zIA5'), 'z5l6'),
                        ((-3648, 'zGDJ'), 'zCE3'),
                    ),
                ),
            ),
        )

        f5 = f1.relabel_shift_out([0, 1], axis=1)
        self.assertEqual(f5.index.name, ('a', 'b'))
        self.assertEqual(
            f5.to_pairs(),
            (
                (
                    0,
                    (
                        ('x', 'zZbu'),
                        ('y', 105269),
                        ((34715, 'zOyq'), 'zjZQ'),
                        ((34715, 'zIA5'), 'zO5l'),
                        ((-3648, 'zGDJ'), 'zEdH'),
                    ),
                ),
                (
                    1,
                    (
                        ('x', 'zZbu'),
                        ('y', 119909),
                        ((34715, 'zOyq'), 'zaji'),
                        ((34715, 'zIA5'), 'zJnC'),
                        ((-3648, 'zGDJ'), 'zDdR'),
                    ),
                ),
                (
                    2,
                    (
                        ('x', 'ztsv'),
                        ('y', 194224),
                        ((34715, 'zOyq'), 'ztsv'),
                        ((34715, 'zIA5'), 'zUvW'),
                        ((-3648, 'zGDJ'), 'zkuW'),
                    ),
                ),
                (
                    3,
                    (
                        ('x', 'ztsv'),
                        ('y', 172133),
                        ((34715, 'zOyq'), 'z2Oo'),
                        ((34715, 'zIA5'), 'z5l6'),
                        ((-3648, 'zGDJ'), 'zCE3'),
                    ),
                ),
            ),
        )

    def test_frame_relabel_shift_out_c(self) -> None:
        f1 = ff.parse('s(3,4)|v(str)').rename(index=('a', 'b'), columns=('x', 'y'))

        with self.assertRaises(AxisInvalid):
            _ = f1.relabel_shift_out(0, axis=2)

        f2 = f1.relabel_shift_out(0, axis=0)
        self.assertEqual(f2.columns.name, ('x', 'y'))

        self.assertEqual(
            f2.to_pairs(),
            (
                (('a', 'b'), ((0, 0), (1, 1), (2, 2))),
                (0, ((0, 'zjZQ'), (1, 'zO5l'), (2, 'zEdH'))),
                (1, ((0, 'zaji'), (1, 'zJnC'), (2, 'zDdR'))),
                (2, ((0, 'ztsv'), (1, 'zUvW'), (2, 'zkuW'))),
                (3, ((0, 'z2Oo'), (1, 'z5l6'), (2, 'zCE3'))),
            ),
        )

        f3 = f1.relabel_shift_out(0, axis=1)
        self.assertEqual(f3.index.name, ('a', 'b'))
        self.assertEqual(
            f3.to_pairs(),
            (
                (0, ((('x', 'y'), 0), (0, 'zjZQ'), (1, 'zO5l'), (2, 'zEdH'))),
                (1, ((('x', 'y'), 1), (0, 'zaji'), (1, 'zJnC'), (2, 'zDdR'))),
                (2, ((('x', 'y'), 2), (0, 'ztsv'), (1, 'zUvW'), (2, 'zkuW'))),
                (3, ((('x', 'y'), 3), (0, 'z2Oo'), (1, 'z5l6'), (2, 'zCE3'))),
            ),
        )

    def test_frame_relabel_shift_out_d(self) -> None:
        f1 = ff.parse('s(3,4)|i(IH,(int,str,int,str))|c(I,str)|v(str)').rename(
            index=('a', 'b'), columns=('x', 'y')
        )
        f2 = f1.relabel_shift_out([0, 1], axis=0)
        self.assertEqual(f2.index.depth, 2)
        self.assertEqual(f2.columns.depth, 1)
        self.assertEqual(
            f2.to_pairs(),
            (
                (
                    '__index0__',
                    (
                        ((58768, 'zoUj'), 34715),
                        ((58768, 'zjDm'), 34715),
                        ((146284, 'zFB6'), 34715),
                    ),
                ),
                (
                    '__index1__',
                    (
                        ((58768, 'zoUj'), 'zOyq'),
                        ((58768, 'zjDm'), 'zOyq'),
                        ((146284, 'zFB6'), 'zOyq'),
                    ),
                ),
                (
                    'zZbu',
                    (
                        ((58768, 'zoUj'), 'zjZQ'),
                        ((58768, 'zjDm'), 'zO5l'),
                        ((146284, 'zFB6'), 'zEdH'),
                    ),
                ),
                (
                    'ztsv',
                    (
                        ((58768, 'zoUj'), 'zaji'),
                        ((58768, 'zjDm'), 'zJnC'),
                        ((146284, 'zFB6'), 'zDdR'),
                    ),
                ),
                (
                    'zUvW',
                    (
                        ((58768, 'zoUj'), 'ztsv'),
                        ((58768, 'zjDm'), 'zUvW'),
                        ((146284, 'zFB6'), 'zkuW'),
                    ),
                ),
                (
                    'zkuW',
                    (
                        ((58768, 'zoUj'), 'z2Oo'),
                        ((58768, 'zjDm'), 'z5l6'),
                        ((146284, 'zFB6'), 'zCE3'),
                    ),
                ),
            ),
        )

    def test_frame_relabel_shift_out_e(self) -> None:
        f1 = ff.parse('f(Fg)|s(3,4)|c(IHg,(int,str,int,str))|i(I,str)|v(str)').rename(
            index=('a', 'b'), columns=('x', 'y')
        )
        f1[(12345, 'abcd', 67890, 'efgh')] = 'ijkl'
        f2 = f1.relabel_shift_out([0, 1], axis=1)
        self.assertEqual(f2.index.depth, 1)
        self.assertEqual(f2.columns.depth, 2)
        self.assertEqual(
            f2.to_pairs(),
            (
                (
                    (58768, 'zoUj'),
                    (
                        ('__index0__', 34715),
                        ('__index1__', 'zOyq'),
                        ('zZbu', 'zjZQ'),
                        ('ztsv', 'zO5l'),
                        ('zUvW', 'zEdH'),
                    ),
                ),
                (
                    (58768, 'zjDm'),
                    (
                        ('__index0__', 34715),
                        ('__index1__', 'zOyq'),
                        ('zZbu', 'zaji'),
                        ('ztsv', 'zJnC'),
                        ('zUvW', 'zDdR'),
                    ),
                ),
                (
                    (146284, 'zFB6'),
                    (
                        ('__index0__', 34715),
                        ('__index1__', 'zOyq'),
                        ('zZbu', 'ztsv'),
                        ('ztsv', 'zUvW'),
                        ('zUvW', 'zkuW'),
                    ),
                ),
                (
                    (146284, 'zneC'),
                    (
                        ('__index0__', 34715),
                        ('__index1__', 'zOyq'),
                        ('zZbu', 'z2Oo'),
                        ('ztsv', 'z5l6'),
                        ('zUvW', 'zCE3'),
                    ),
                ),
                (
                    (67890, 'efgh'),
                    (
                        ('__index0__', 12345),
                        ('__index1__', 'abcd'),
                        ('zZbu', 'ijkl'),
                        ('ztsv', 'ijkl'),
                        ('zUvW', 'ijkl'),
                    ),
                ),
            ),
        )

    def test_frame_relabel_shift_out_f(self) -> None:
        msg = (
            'date,currency,price',
            '2022-01-01,USD,2.48',
            '2022-01-02,USD,2.52',
            '2022-01-03,USD,2.49',
            '2022-01-04,USD,2.51',
        )
        f1 = Frame.from_csv(
            msg,
            dtypes=['datetime64[D]', str, float],
            index_depth=2,
            index_constructors=[IndexDate, Index],
        )
        f2 = f1.relabel_shift_out(1)  # ErrorInitIndex

        self.assertEqual(
            f2.to_pairs(),
            (
                (
                    '__index1__',
                    (
                        (np.datetime64('2022-01-01'), 'USD'),
                        (np.datetime64('2022-01-02'), 'USD'),
                        (np.datetime64('2022-01-03'), 'USD'),
                        (np.datetime64('2022-01-04'), 'USD'),
                    ),
                ),
                (
                    'price',
                    (
                        (np.datetime64('2022-01-01'), 2.48),
                        (np.datetime64('2022-01-02'), 2.52),
                        (np.datetime64('2022-01-03'), 2.49),
                        (np.datetime64('2022-01-04'), 2.51),
                    ),
                ),
            ),
        )

    # ---------------------------------------------------------------------------

    def test_frame_rank_a(self) -> None:
        f = ff.parse('s(4,6)|v(int)|i(I,str)')

        self.assertEqual(
            f._rank(method='ordinal', axis=0).to_pairs(),
            (
                (0, (('zZbu', 0), ('ztsv', 3), ('zUvW', 2), ('zkuW', 1))),
                (1, (('zZbu', 3), ('ztsv', 1), ('zUvW', 2), ('zkuW', 0))),
                (2, (('zZbu', 0), ('ztsv', 3), ('zUvW', 1), ('zkuW', 2))),
                (3, (('zZbu', 2), ('ztsv', 0), ('zUvW', 3), ('zkuW', 1))),
                (4, (('zZbu', 1), ('ztsv', 2), ('zUvW', 3), ('zkuW', 0))),
                (5, (('zZbu', 2), ('ztsv', 0), ('zUvW', 3), ('zkuW', 1))),
            ),
        )

        self.assertEqual(
            f._rank(method='ordinal', axis=1).to_pairs(),
            (
                (0, (('zZbu', 0), ('ztsv', 4), ('zUvW', 2), ('zkuW', 1))),
                (1, (('zZbu', 5), ('ztsv', 0), ('zUvW', 0), ('zkuW', 0))),
                (2, (('zZbu', 1), ('ztsv', 3), ('zUvW', 1), ('zkuW', 3))),
                (3, (('zZbu', 4), ('ztsv', 2), ('zUvW', 3), ('zkuW', 5))),
                (4, (('zZbu', 2), ('ztsv', 5), ('zUvW', 4), ('zkuW', 2))),
                (5, (('zZbu', 3), ('ztsv', 1), ('zUvW', 5), ('zkuW', 4))),
            ),
        )

        self.assertEqual(
            f._rank(method='ordinal', axis=0, ascending=False).to_pairs(),
            (
                (0, (('zZbu', 3), ('ztsv', 0), ('zUvW', 1), ('zkuW', 2))),
                (1, (('zZbu', 0), ('ztsv', 2), ('zUvW', 1), ('zkuW', 3))),
                (2, (('zZbu', 3), ('ztsv', 0), ('zUvW', 2), ('zkuW', 1))),
                (3, (('zZbu', 1), ('ztsv', 3), ('zUvW', 0), ('zkuW', 2))),
                (4, (('zZbu', 2), ('ztsv', 1), ('zUvW', 0), ('zkuW', 3))),
                (5, (('zZbu', 1), ('ztsv', 3), ('zUvW', 0), ('zkuW', 2))),
            ),
        )

        self.assertEqual(
            f._rank(method='ordinal', axis=1, ascending=False).to_pairs(),
            (
                (0, (('zZbu', 5), ('ztsv', 1), ('zUvW', 3), ('zkuW', 4))),
                (1, (('zZbu', 0), ('ztsv', 5), ('zUvW', 5), ('zkuW', 5))),
                (2, (('zZbu', 4), ('ztsv', 2), ('zUvW', 4), ('zkuW', 2))),
                (3, (('zZbu', 1), ('ztsv', 3), ('zUvW', 2), ('zkuW', 0))),
                (4, (('zZbu', 3), ('ztsv', 0), ('zUvW', 1), ('zkuW', 3))),
                (5, (('zZbu', 2), ('ztsv', 4), ('zUvW', 0), ('zkuW', 1))),
            ),
        )

    def test_frame_rank_b(self) -> None:
        f = Frame.from_fields(
            [[0, 0, 1], [0, 0, 1]], index=('a', 'b', 'c'), columns=('x', 'y')
        )
        self.assertEqual(
            f.rank_ordinal(ascending=[True, False], start=1).to_pairs(),
            (
                ('x', (('a', 1), ('b', 2), ('c', 3))),
                ('y', (('a', 3), ('b', 2), ('c', 1))),
            ),
        )
        self.assertEqual(
            f.rank_dense(ascending=[True, False], start=1).to_pairs(),
            (
                ('x', (('a', 1), ('b', 1), ('c', 2))),
                ('y', (('a', 2), ('b', 2), ('c', 1))),
            ),
        )
        self.assertEqual(
            f.rank_mean(ascending=[True, False], start=1).to_pairs(),
            (
                ('x', (('a', 1.5), ('b', 1.5), ('c', 3.0))),
                ('y', (('a', 2.5), ('b', 2.5), ('c', 1.0))),
            ),
        )
        self.assertEqual(
            f.rank_min(ascending=[True, False], start=0).to_pairs(),
            (
                ('x', (('a', 0), ('b', 0), ('c', 2))),
                ('y', (('a', 1), ('b', 1), ('c', 0))),
            ),
        )

        self.assertEqual(
            f.rank_max(ascending=[True, False], start=0).to_pairs(),
            (
                ('x', (('a', 1), ('b', 1), ('c', 2))),
                ('y', (('a', 2), ('b', 2), ('c', 0))),
            ),
        )

    def test_frame_rank_c(self) -> None:
        f1 = Frame.from_fields(
            [[np.nan, 0, 1], [0, None, 1]], index=('a', 'b', 'c'), columns=('x', 'y')
        )

        f2 = f1.rank_ordinal(axis=0)
        self.assertEqual(f2.values.dtype, float)
        f3 = f1.rank_ordinal(axis=1)
        self.assertEqual(f3.values.dtype, float)

        f4 = f1.rank_ordinal(axis=0, fill_value=-1)
        self.assertEqual(f4.values.dtype.kind, 'i')
        f5 = f1.rank_ordinal(axis=1, fill_value=-1)
        self.assertEqual(f5.values.dtype.kind, 'i')

    def test_frame_rank_c1(self) -> None:
        f1 = Frame.from_fields(
            [[np.nan, 0, 1], [0, None, 1], [2, None, -1]],
            index=('a', 'b', 'c'),
            columns=('x', 'y', 'z'),
        )

        f2 = f1.rank_ordinal(axis=0, fill_value=[-1, -2, -3])  # rank within
        self.assertEqual(
            f2.to_pairs(),
            (
                ('x', (('a', -1), ('b', 0), ('c', 1))),
                ('y', (('a', 0), ('b', -2), ('c', 1))),
                ('z', (('a', 1), ('b', -3), ('c', 0))),
            ),
        )
        with self.assertRaises(InvalidFillValue):
            f1.rank_ordinal(axis=1, fill_value=FillValueAuto)  # rank within rows

    def test_frame_rank_d(self) -> None:
        f1 = Frame.from_fields(
            [[np.nan, 0, 1], [0, None, 1]], index=('a', 'b', 'c'), columns=('x', 'y')
        )
        with self.assertRaises(AxisInvalid):
            f1.rank_ordinal(axis=3)

    def test_frame_rank_ordinal(self) -> None:
        f1 = sf.Frame.from_records(
            [[8, 15, 7, 2, 20, 4, 20, 7, 15, 15], [5, 6, 3, 1, 9, 2, 9, 3, 6, 6]]
        )
        f2 = f1.rank_ordinal()
        self.assertEqual(
            f2.to_pairs(),
            (
                (0, ((0, 1), (1, 0))),
                (1, ((0, 1), (1, 0))),
                (2, ((0, 1), (1, 0))),
                (3, ((0, 1), (1, 0))),
                (4, ((0, 1), (1, 0))),
                (5, ((0, 1), (1, 0))),
                (6, ((0, 1), (1, 0))),
                (7, ((0, 1), (1, 0))),
                (8, ((0, 1), (1, 0))),
                (9, ((0, 1), (1, 0))),
            ),
        )

        f3 = f1.rank_ordinal(ascending=(v % 2 for v in f1.iloc[0].values))
        self.assertEqual(
            f3.to_pairs(),
            (
                (0, ((0, 0), (1, 1))),
                (1, ((0, 1), (1, 0))),
                (2, ((0, 1), (1, 0))),
                (3, ((0, 0), (1, 1))),
                (4, ((0, 0), (1, 1))),
                (5, ((0, 0), (1, 1))),
                (6, ((0, 0), (1, 1))),
                (7, ((0, 1), (1, 0))),
                (8, ((0, 1), (1, 0))),
                (9, ((0, 1), (1, 0))),
            ),
        )

        f4 = f1.rank_ordinal(axis=1)
        self.assertEqual(
            f4.to_pairs(),
            (
                (0, ((0, 4), (1, 4))),
                (1, ((0, 5), (1, 5))),
                (2, ((0, 2), (1, 2))),
                (3, ((0, 0), (1, 0))),
                (4, ((0, 8), (1, 8))),
                (5, ((0, 1), (1, 1))),
                (6, ((0, 9), (1, 9))),
                (7, ((0, 3), (1, 3))),
                (8, ((0, 6), (1, 6))),
                (9, ((0, 7), (1, 7))),
            ),
        )

        with self.assertRaises(RuntimeError):
            _ = f1.rank_ordinal(axis=1, ascending=(False, True, False))

    def test_frame_rank_mean(self) -> None:
        f1 = sf.Frame.from_records(
            [[8, 15, 7, 2, 20, 4, 20, 7, 15, 15], [5, 6, 3, 1, 9, 2, 9, 3, 6, 6]]
        )
        f2 = f1.rank_mean()
        self.assertEqual(
            f2.to_pairs(),
            (
                (0, ((0, 1.0), (1, 0.0))),
                (1, ((0, 1.0), (1, 0.0))),
                (2, ((0, 1.0), (1, 0.0))),
                (3, ((0, 1.0), (1, 0.0))),
                (4, ((0, 1.0), (1, 0.0))),
                (5, ((0, 1.0), (1, 0.0))),
                (6, ((0, 1.0), (1, 0.0))),
                (7, ((0, 1.0), (1, 0.0))),
                (8, ((0, 1.0), (1, 0.0))),
                (9, ((0, 1.0), (1, 0.0))),
            ),
        )
        f3 = f1.rank_mean(axis=1)
        self.assertEqual(
            f3.to_pairs(),
            (
                (0, ((0, 4.0), (1, 4.0))),
                (1, ((0, 6.0), (1, 6.0))),
                (2, ((0, 2.5), (1, 2.5))),
                (3, ((0, 0.0), (1, 0.0))),
                (4, ((0, 8.5), (1, 8.5))),
                (5, ((0, 1.0), (1, 1.0))),
                (6, ((0, 8.5), (1, 8.5))),
                (7, ((0, 2.5), (1, 2.5))),
                (8, ((0, 6.0), (1, 6.0))),
                (9, ((0, 6.0), (1, 6.0))),
            ),
        )

    def test_frame_rank_min(self) -> None:
        f1 = sf.Frame.from_records(
            [[8, 15, 7, 2, 20, 4, 20, 7, 15, 15], [5, 6, 3, 1, 9, 2, 9, 3, 6, 6]]
        )
        f2 = f1.rank_min(axis=1)
        self.assertEqual(
            f2.to_pairs(),
            (
                (0, ((0, 4), (1, 4))),
                (1, ((0, 5), (1, 5))),
                (2, ((0, 2), (1, 2))),
                (3, ((0, 0), (1, 0))),
                (4, ((0, 8), (1, 8))),
                (5, ((0, 1), (1, 1))),
                (6, ((0, 8), (1, 8))),
                (7, ((0, 2), (1, 2))),
                (8, ((0, 5), (1, 5))),
                (9, ((0, 5), (1, 5))),
            ),
        )

    def test_frame_rank_max(self) -> None:
        f1 = sf.Frame.from_records(
            [[8, 15, 7, 2, 20, 4, 20, 7, 15, 15], [5, 6, 3, 1, 9, 2, 9, 3, 6, 6]]
        )
        f2 = f1.rank_max(axis=1)
        self.assertEqual(
            f2.to_pairs(),
            (
                (0, ((0, 4), (1, 4))),
                (1, ((0, 7), (1, 7))),
                (2, ((0, 3), (1, 3))),
                (3, ((0, 0), (1, 0))),
                (4, ((0, 9), (1, 9))),
                (5, ((0, 1), (1, 1))),
                (6, ((0, 9), (1, 9))),
                (7, ((0, 3), (1, 3))),
                (8, ((0, 7), (1, 7))),
                (9, ((0, 7), (1, 7))),
            ),
        )

    def test_frame_rank_dense(self) -> None:
        f1 = sf.Frame.from_records(
            [[8, 15, 7, 2, 20, 4, 20, 7, 15, 15], [5, 6, 3, 1, 9, 2, 9, 3, 6, 6]]
        )
        f2 = f1.rank_dense(axis=1)
        self.assertEqual(
            f2.to_pairs(),
            (
                (0, ((0, 3), (1, 3))),
                (1, ((0, 4), (1, 4))),
                (2, ((0, 2), (1, 2))),
                (3, ((0, 0), (1, 0))),
                (4, ((0, 5), (1, 5))),
                (5, ((0, 1), (1, 1))),
                (6, ((0, 5), (1, 5))),
                (7, ((0, 2), (1, 2))),
                (8, ((0, 4), (1, 4))),
                (9, ((0, 4), (1, 4))),
            ),
        )

    # ---------------------------------------------------------------------------

    def test_frame_zero_size_a(self) -> None:
        f1 = Frame.from_element(0, index=range(3), columns=[0])
        f2 = f1.drop[0]
        self.assertEqual(f2.shape, (3, 0))
        self.assertEqual(f2.index.values.tolist(), [0, 1, 2])
        # slicing by rows should return a Frame of shape (2, 0)
        f3 = f2.loc[1:]
        self.assertEqual(f3.shape, (2, 0))
        self.assertEqual(f3.index.values.tolist(), [1, 2])
        self.assertEqual(f2.loc[[0, 2]].index.values.tolist(), [0, 2])

    # ---------------------------------------------------------------------------

    def test_frame_assign_apply_index_hierarchy_columns(self) -> None:
        f = Frame.from_element(1, index=IndexHierarchy.from_labels([(0, 0)]), columns=[0])
        post = f.assign[0].apply(lambda x: x * 2)

        expected = Frame.from_element(2, index=f.index, columns=f.columns)
        self.assertTrue(post.equals(expected))

    # ---------------------------------------------------------------------------

    def test_frame_assign_apply_element_a(self) -> None:
        f1 = ff.parse('s(2,3)|v(str)|i(I,str)')
        f2 = f1.assign[1:].apply_element(lambda e: '' if 'a' in e or 'v' in e else e)
        # f2 = f1.assign[1] + -f1[1]

        self.assertEqual(
            f2.to_pairs(),
            (
                (0, (('zZbu', 'zjZQ'), ('ztsv', 'zO5l'))),
                (1, (('zZbu', ''), ('ztsv', 'zJnC'))),
                (2, (('zZbu', ''), ('ztsv', ''))),
            ),
        )

    def test_frame_assign_apply_element_b(self) -> None:
        f1 = Frame.from_records(
            [
                ['2023-01-01', '2023-02-01'],
                ['2023-01-02', '2023-02-02'],
                ['2023-01-03', '2023-02-03'],
            ],
            columns=['A', 'B'],
        )

        f2 = f1.assign['A'].apply_element(
            datetime.date.fromisoformat, dtype='datetime64[D]'
        )
        self.assertEqual(
            f2.dtypes.values.tolist(), [np.dtype('<M8[D]'), np.dtype('<U10')]
        )

        f3 = f1.assign[['A']].apply_element(
            datetime.date.fromisoformat, dtype='datetime64[D]'
        )
        self.assertEqual(
            f3.dtypes.values.tolist(), [np.dtype('<M8[D]'), np.dtype('<U10')]
        )

    def test_frame_assign_apply_element_items_a(self) -> None:
        f1 = ff.parse('s(2,3)|v(str)|i(I,str)')
        f2 = f1.assign.loc['ztsv'].apply_element_items(
            lambda label, e: label if label < 2 else e
        )
        # f2 = f1.assign[1:].apply_element(lambda e: '' if 'a' in e or 'v' in e else e)
        self.assertEqual(
            f2.to_pairs(),
            (
                (0, (('zZbu', 'zjZQ'), ('ztsv', 0))),
                (1, (('zZbu', 'zaji'), ('ztsv', 1))),
                (2, (('zZbu', 'ztsv'), ('ztsv', 'zUvW'))),
            ),
        )

    def test_frame_assign_apply_element_items_b(self) -> None:
        f1 = ff.parse('s(2,3)|v(str)|i(I,str)')
        f2 = f1.assign[1:].apply_element_items(lambda label, e: label)
        self.assertEqual(
            f2.to_pairs(),
            (
                (0, (('zZbu', 'zjZQ'), ('ztsv', 'zO5l'))),
                (1, (('zZbu', ('zZbu', 1)), ('ztsv', ('ztsv', 1)))),
                (2, (('zZbu', ('zZbu', 2)), ('ztsv', ('ztsv', 2)))),
            ),
        )

    def test_frame_assign_apply_element_items_c(self) -> None:
        f1 = ff.parse('s(4,4)|v(int64)|i(I,str)')
        f2 = f1.assign.bloc[f1 < 0].apply_element(lambda e: 0 if e < -40000 else e)
        self.assertEqual(
            f2.to_pairs(),
            (
                (0, (('zZbu', 0), ('ztsv', 92867), ('zUvW', 84967), ('zkuW', 13448))),
                (1, (('zZbu', 162197), ('ztsv', 0), ('zUvW', 5729), ('zkuW', 0))),
                (
                    2,
                    (
                        ('zZbu', -3648),
                        ('ztsv', 91301),
                        ('zUvW', 30205),
                        ('zkuW', 54020),
                    ),
                ),
                (
                    3,
                    (
                        ('zZbu', 129017),
                        ('ztsv', 35021),
                        ('zUvW', 166924),
                        ('zkuW', 122246),
                    ),
                ),
            ),
        )

    # ---------------------------------------------------------------------------
    def test_frame_name_assign_index_hierarchy_a(self) -> None:
        f = ff.parse('f(Fg)|v(int,bool,str)|i((IY,ID),(dtY,dtD))|c(Isg,dts)|s(3,2)')
        post = f.loc[sf.HLoc[f.index.iloc[0]]].name
        self.assertEqual(post, (np.datetime64('36685', 'Y'), np.datetime64('2258-03-21')))

    def test_frame_name_assign_index_hierarchy_b(self) -> None:
        f = FrameGO.from_element(
            0,
            index=(0, 1),
            columns=IndexHierarchy.from_labels((('a', 0), ('a', 1))),
        )
        f[('b', 0)] = 1
        f[('b', 1)] = 1
        post = f.columns._extract_iloc_by_int(3)
        self.assertEqual(post, ('b', 1))
        s = f[('b', 1)]
        self.assertEqual(s.name, ('b', 1))

    # ---------------------------------------------------------------------------
    def test_frame_bloc_sel_a(self) -> None:
        # this proves row-major ordering and not sorting on hashables
        f = Frame(np.arange(6).reshape(3, 2), index=(3, 2, 1), columns=('z', None))
        self.assertEqual(
            f.bloc[f >= 0].to_pairs(),
            (
                ((3, 'z'), 0),
                ((3, None), 1),
                ((2, 'z'), 2),
                ((2, None), 3),
                ((1, 'z'), 4),
                ((1, None), 5),
            ),
        )

    # ---------------------------------------------------------------------------
    def test_frame_dfi_a(self) -> None:
        f = ff.parse('f(Fg)|v(int,bool,str)|c(Ig,str)|s(4,8)')
        dfi = f.__dataframe__()
        self.assertEqual(dfi.num_columns(), 8)
        self.assertEqual(dfi.num_rows(), 4)

    # ---------------------------------------------------------------------------
    def test_frame_to_json_index_a(self) -> None:
        f = ff.parse('s(2,4)|v(bool,int,float,str)|i(I,int)|c(I,str)')
        post = f.to_json_index()
        self.assertEqual(
            post,
            '{"34715": {"zZbu": false, "ztsv": 162197, "zUvW": 694.3, "zkuW": "z2Oo"}, "-3648": {"zZbu": false, "ztsv": -41157, "zUvW": -72.96, "zkuW": "z5l6"}}',
        )

    # ---------------------------------------------------------------------------
    def test_frame_to_json_columns_a(self) -> None:
        f = ff.parse('s(2,4)|v(bool,int,float,str)|i(I,int)|c(I,str)')
        post = f.to_json_columns()
        self.assertEqual(
            post,
            '{"zZbu": {"34715": false, "-3648": false}, "ztsv": {"34715": 162197, "-3648": -41157}, "zUvW": {"34715": 694.3, "-3648": -72.96}, "zkuW": {"34715": "z2Oo", "-3648": "z5l6"}}',
        )

    # ---------------------------------------------------------------------------
    def test_frame_to_json_split_a(self) -> None:
        f = ff.parse('s(2,4)|v(bool,int,float,str)|i(I,int)|c(I,str)')
        post = f.to_json_split()
        self.assertEqual(
            post,
            '{"columns": ["zZbu", "ztsv", "zUvW", "zkuW"], "index": [34715, -3648], "data": [[false, 162197, 694.3, "z2Oo"], [false, -41157, -72.96, "z5l6"]]}',
        )

    # ---------------------------------------------------------------------------

    def test_frame_to_json_typed_a(self) -> None:
        f = ff.parse('s(2,4)|v(bool,int32,float,str)|i(I,int32)|c(I,str)')
        post = f.to_json_typed()

        self.assertEqual(
            post,
            '{"columns": ["zZbu", "ztsv", "zUvW", "zkuW"], "index": [34715, -3648], "data": [[false, false], [162197, -41157], [694.3, -72.96], ["z2Oo", "z5l6"]], "__meta__": {"__names__": [null, null, null], "__dtypes__": ["|b1", "=i4", "=f8", "=U4"], "__dtypes_index__": ["=i4"], "__dtypes_columns__": ["=U4"], "__types__": ["Index", "Index"], "__depths__": [4, 1, 1]}}',
        )

    def test_frame_to_json_typed_b(self) -> None:
        f = ff.parse('s(2,2)|v(bool)')
        post = f.to_json_typed()
        self.assertEqual(
            post,
            '{"columns": null, "index": null, "data": [[false, false], [false, false]], "__meta__": {"__names__": [null, null, null], "__dtypes__": ["|b1", "|b1"], "__dtypes_index__": ["=i8"], "__dtypes_columns__": ["=i8"], "__types__": ["Index", "Index"], "__depths__": [2, 1, 1]}}',
        )

    # ---------------------------------------------------------------------------
    def test_frame_to_json_records_a(self) -> None:
        f = ff.parse('s(2,4)|v(bool,int,float,str)|i(I,int)|c(I,str)')
        post = f.to_json_records()
        self.assertEqual(
            post,
            '[{"zZbu": false, "ztsv": 162197, "zUvW": 694.3, "zkuW": "z2Oo"}, {"zZbu": false, "ztsv": -41157, "zUvW": -72.96, "zkuW": "z5l6"}]',
        )

    # ---------------------------------------------------------------------------
    def test_frame_to_json_values_a(self) -> None:
        f = ff.parse('s(2,4)|v(bool,int,float,str)|i(I,int)|c(I,str)')
        post = f.to_json_values()
        self.assertEqual(
            post, '[[false, 162197, 694.3, "z2Oo"], [false, -41157, -72.96, "z5l6"]]'
        )

    # ---------------------------------------------------------------------------
    def test_frame_from_json_index_a(self) -> None:
        f1 = ff.parse('s(2,4)|v(bool,int,float,str)|i(I,int)|c(I,str)')
        post = f1.to_json_index()
        f2 = Frame.from_json_index(post, index_constructor=partial(sf.Index, dtype=int))
        self.assertTrue(f1.equals(f2))

    def test_frame_from_json_index_b(self) -> None:
        f1 = ff.parse('s(2,4)|v(bool,int,float,str)|i(I,int)|c(I,str)')
        post = f1.to_json_index()
        f2 = Frame.from_json_index(
            StringIO(post), index_constructor=partial(sf.Index, dtype=int)
        )
        self.assertTrue(f1.equals(f2))

    def test_frame_from_json_columns_a(self) -> None:
        f1 = ff.parse('s(2,4)|v(bool,int,float,str)|i(I,int)|c(I,str)')
        post = f1.to_json_columns()
        f2 = Frame.from_json_columns(post, index_constructor=partial(sf.Index, dtype=int))
        self.assertTrue(f1.equals(f2))

    def test_frame_from_json_columns_b(self) -> None:
        f1 = ff.parse('s(2,4)|v(bool,int,float,str)|i(I,int)|c(I,str)')
        post = f1.to_json_columns()
        f2 = Frame.from_json_columns(
            StringIO(post), index_constructor=partial(sf.Index, dtype=int)
        )
        self.assertTrue(f1.equals(f2))

    # ---------------------------------------------------------------------------

    def test_frame_from_json_split_a(self) -> None:
        f1 = ff.parse('s(2,4)|v(bool,int,float,str)|i(I,int)|c(I,str)')
        post = f1.to_json_split()
        f2 = Frame.from_json_split(post, index_constructor=partial(sf.Index, dtype=int))
        self.assertTrue(f1.equals(f2))

    def test_frame_from_json_split_b(self) -> None:
        f1 = ff.parse('s(2,4)|v(bool,int,float,str)|i(I,int)|c(I,str)')
        post = f1.to_json_split()
        f2 = Frame.from_json_split(
            StringIO(post), index_constructor=partial(sf.Index, dtype=int)
        )
        self.assertTrue(f1.equals(f2))

    # ---------------------------------------------------------------------------

    def test_frame_from_json_typed_a(self) -> None:
        f1 = ff.parse('s(2,4)|v(int8,str)|i(IH,(str,str))|c(IH,(int64,int64))').rename(
            'x', index='y', columns='z'
        )
        post1 = f1.to_json_typed()

        self.assertEqual(
            post1,
            '{"columns": [[34715, 105269], [34715, 119909], [-3648, 194224], [-3648, 172133]], "index": [["zZbu", "zOyq"], ["zZbu", "zIA5"]], "data": [[47, -61], ["zaji", "zJnC"], [-64, -91], ["z2Oo", "z5l6"]], "__meta__": {"__names__": ["x", "y", "z"], "__dtypes__": ["|i1", "=U4", "|i1", "=U4"], "__dtypes_index__": ["=U4", "=U4"], "__dtypes_columns__": ["=i8", "=i8"], "__types__": ["IndexHierarchy", "IndexHierarchy"], "__types_index__": ["Index", "Index"], "__types_columns__": ["Index", "Index"], "__depths__": [4, 2, 2]}}',
        )

        f2 = Frame.from_json_typed(
            post1,
        )

        self.assertEqual(f2.to_json_typed(), post1)

        self.assertTrue(
            f1.equals(f2, compare_name=True, compare_dtype=True, compare_class=True)
        )

    def test_frame_from_json_typed_b(self) -> None:
        f1 = ff.parse(
            's(2,4)|v(int8,str)|i((ID, I),(dtD,str))|c((Is, I),(dts,int64))'
        ).rename('x', index='y', columns='z')
        post1 = f1.to_json_typed()

        f2 = Frame.from_json_typed(post1)
        self.assertEqual(f2.to_json_typed(), post1)

        self.assertTrue(
            f1.equals(f2, compare_name=True, compare_dtype=True, compare_class=True)
        )

    def test_frame_from_json_typed_c(self) -> None:
        f1 = ff.parse('s(2,4)|v(bool,int32,float,str)|i(I,int32)|c(I,str)').rename(
            'x', index='y', columns='z'
        )
        post = f1.to_json_typed()
        f2 = Frame.from_json_typed(post)
        self.assertTrue(
            f1.equals(f2, compare_name=True, compare_dtype=True, compare_class=True)
        )

    def test_frame_from_json_typed_d(self) -> None:
        f1 = ff.parse('s(2,4)|v(int8,str)|i(ID,dtD)|c(ID,dtD)').rename(
            'x', index='y', columns='z'
        )
        post = f1.to_json_typed()
        f2 = Frame.from_json_typed(post)
        self.assertTrue(
            f1.equals(f2, compare_name=True, compare_dtype=True, compare_class=True)
        )

    def test_frame_from_json_typed_e(self) -> None:
        f1 = ff.parse('s(2,4)|v(int8,str)|i(ID,dtD)|c(ID,dtD)').rename(
            'x', index='y', columns='z'
        )
        post = f1.to_json_typed()
        f2 = FrameGO.from_json_typed(post)
        self.assertTrue(
            f1.equals(f2, compare_name=True, compare_dtype=True, compare_class=False)
        )

    def test_frame_from_json_typed_f(self) -> None:
        f1 = ff.parse('s(2,4)|v(int8,str)|i(ID,dtD)|c(ID,dtD)').rename(
            'x', index='y', columns='z'
        )
        # <Frame: x>
        # <IndexDate: z>  2065-01-17 1979-12-28 2219-12-23 2052-09-12 <datetime64[D]>
        # <IndexDate: y>
        # 2065-01-17      47         zaji       -64        z2Oo
        # 1979-12-28      -61        zJnC       -91        z5l6
        # <datetime64[D]> <int8>     <<U4>      <int8>     <<U4>
        post = f1.to_json_typed()
        sio = io.StringIO(post)
        f2 = FrameGO.from_json_typed(sio)
        self.assertTrue(
            f1.equals(f2, compare_name=True, compare_dtype=True, compare_class=False)
        )

    def test_frame_from_json_typed_g(self) -> None:
        f1 = ff.parse('s(2,3)|v(str,bool)').rename('x', index='y', columns='z')
        post = f1.to_json_typed()
        f2 = Frame.from_json_typed(post)
        self.assertIsNone(f2._index._map)
        self.assertIsNone(f2._columns._map)
        self.assertTrue(
            f1.equals(f2, compare_name=True, compare_dtype=True, compare_class=True)
        )

    # ---------------------------------------------------------------------------

    def test_frame_from_json_records_a(self) -> None:
        f1 = ff.parse('s(2,4)|v(bool,int,float,str)|c(I,str)')
        post = f1.to_json_records()
        f2 = Frame.from_json_records(post)
        self.assertTrue(f1.equals(f2))

    def test_frame_from_json_records_b(self) -> None:
        f1 = ff.parse('s(2,4)|v(bool,int,float,str)|c(I,str)')
        post = f1.to_json_records()
        f2 = Frame.from_json_records(StringIO(post))
        self.assertTrue(f1.equals(f2))

    def test_frame_from_json_values_a(self) -> None:
        f1 = ff.parse('s(2,4)|v(bool,int,float,str)')
        post = f1.to_json_values()
        f2 = Frame.from_json_values(post)
        self.assertTrue(f1.equals(f2))

    def test_frame_from_json_values_b(self) -> None:
        f1 = ff.parse('s(2,4)|v(bool,int,float,str)')
        post = f1.to_json_values()
        f2 = Frame.from_json_values(StringIO(post))
        self.assertTrue(f1.equals(f2))

    # ---------------------------------------------------------------------------
    def test_frame_from_dict_fields_a1(self) -> None:
        f = Frame.from_dict_fields(
            (dict(a=3, c=4), dict(b=20, c=22), dict(a=120, d=2000)),
            columns=('x', 'y', 'z'),
            fill_value=0,
        )
        self.assertEqual(
            f.to_pairs(),
            (
                ('x', (('a', 3), ('c', 4), ('b', 0), ('d', 0))),
                ('y', (('a', 0), ('c', 22), ('b', 20), ('d', 0))),
                ('z', (('a', 120), ('c', 0), ('b', 0), ('d', 2000))),
            ),
        )

    def test_frame_from_dict_fields_a2(self) -> None:
        f = Frame.from_dict_fields(
            (dict(a=3, c=4), dict(b=20, c=22), dict(a=120, d=2000)),
            columns=('x', 'y', 'z'),
            fill_value=0,
            consolidate_blocks=True,
        )
        self.assertTrue(f._blocks.unified)

    def test_frame_from_dict_fields_b(self) -> None:
        f = Frame.from_dict_fields(
            (dict(a=3, c=4), dict(b=True, c=False), dict(a='w', d='p')),
            columns=('x', 'y', 'z'),
            fill_value=[-1, False, ''],
        )
        self.assertEqual(
            f.to_pairs(),
            (
                ('x', (('a', 3), ('c', 4), ('b', -1), ('d', -1))),
                ('y', (('a', False), ('c', False), ('b', True), ('d', False))),
                ('z', (('a', 'w'), ('c', ''), ('b', ''), ('d', 'p'))),
            ),
        )

    def test_frame_from_dict_fields_c(self) -> None:
        f = Frame.from_dict_fields(
            (dict(a=3, c=4), dict(b=True, c=False), dict(a='w', d='p')),
            columns=('x', 'y', 'z'),
            fill_value=[-1, False, ''],
            dtypes=[int, bool, str],
        )
        self.assertEqual(
            f.to_pairs(),
            (
                ('x', (('a', 3), ('c', 4), ('b', -1), ('d', -1))),
                ('y', (('a', False), ('c', False), ('b', True), ('d', False))),
                ('z', (('a', 'w'), ('c', ''), ('b', ''), ('d', 'p'))),
            ),
        )

    def test_frame_from_dict_fields_d(self) -> None:
        f = Frame.from_dict_fields(
            (dict(a=3, c=4), dict(b=True, c=False), dict(a='w', d='p')),
            fill_value=[-1, False, ''],
            dtypes=[int, bool, str],
        )
        self.assertEqual(
            f.to_pairs(),
            (
                (0, (('a', 3), ('c', 4), ('b', -1), ('d', -1))),
                (1, (('a', False), ('c', False), ('b', True), ('d', False))),
                (2, (('a', 'w'), ('c', ''), ('b', ''), ('d', 'p'))),
            ),
        )

    def test_frame_from_dict_fields_e(self) -> None:
        with self.assertRaises(ErrorInitFrame):
            _ = Frame.from_dict_fields(
                (),
            )

    # ---------------------------------------------------------------------------
    def test_frame_to_signature_bytes_a(self) -> None:
        f1 = ff.parse('f(Fg)|v(int64,bool,str)|c(Ig,str)|s(4,8)')
        b1 = f1._to_signature_bytes(include_name=False)
        self.assertEqual(
            sha256(b1).hexdigest(),
            'a9be99c9d2ab6f60294f2931bc875833993ce3f4d41d8da16802135e041317b6',
        )

        f2 = ff.parse('f(F)|v(int64,bool,str)|c(I,str)|s(4,8)')
        b2 = f2._to_signature_bytes(include_name=False)
        self.assertNotEqual(sha256(b1).hexdigest(), sha256(b2).hexdigest())

    def test_frame_to_signature_bytes_b(self) -> None:
        f1 = ff.parse('f(Fg)|v(int,bool,str)|c(Ig,str)|s(4,8)')
        b1 = f1._to_signature_bytes(include_name=False, include_class=False)

        f2 = ff.parse('f(F)|v(int,bool,str)|c(I,str)|s(4,8)')
        b2 = f2._to_signature_bytes(include_name=False, include_class=False)
        self.assertEqual(sha256(b1).hexdigest(), sha256(b2).hexdigest())

    def test_frame_to_signature_bytes_c(self) -> None:
        f1 = ff.parse('f(F)|v(int,bool,str)|c(I,str)|s(4,8)')
        b1 = f1._to_signature_bytes(include_name=False)

        f2 = ff.parse('f(F)|v(int,bool,str)|c(I,str)|s(4,8)')
        b2 = f2._to_signature_bytes(include_name=False)
        self.assertEqual(sha256(b1).hexdigest(), sha256(b2).hexdigest())

    def test_frame_to_signature_bytes_d(self) -> None:
        f1 = ff.parse('f(F)|v(int,bool,str)|c(I,str)|s(4,8)')
        b1 = f1._to_signature_bytes(include_name=False)

        f2 = ff.parse('f(F)|v(int,bool,str)|c(I,str)|s(3,8)')
        b2 = f2._to_signature_bytes(include_name=False)
        self.assertNotEqual(sha256(b1).hexdigest(), sha256(b2).hexdigest())

    def test_frame_to_signature_bytes_e(self) -> None:
        f1 = ff.parse('f(F)|v(int,bool,str)|c(I,str)|s(4,8)')
        b1 = f1._to_signature_bytes(include_name=False)

        f2 = ff.parse('f(F)|v(int,bool,str)|c(I,int)|s(4,8)')
        b2 = f2._to_signature_bytes(include_name=False)
        self.assertNotEqual(sha256(b1).hexdigest(), sha256(b2).hexdigest())

    def test_frame_to_signature_bytes_f(self) -> None:
        f1 = ff.parse('f(F)|v(int,bool,str)|c(I,str)|s(4,8)')
        b1 = f1._to_signature_bytes(include_name=False)

        f2 = ff.parse('f(F)|v(int,bool,str)|c(I,str)|s(4,8)|i(I,int)')
        b2 = f2._to_signature_bytes(include_name=False)
        self.assertNotEqual(sha256(b1).hexdigest(), sha256(b2).hexdigest())

    # ---------------------------------------------------------------------------

    def test_frame_via_hashlib_a(self) -> None:
        f1 = ff.parse('f(Fg)|v(int64,bool,str)|c(Ig,str)|s(4,8)')
        hd = f1.via_hashlib(include_name=False).sha256().hexdigest()
        self.assertEqual(
            hd, 'a9be99c9d2ab6f60294f2931bc875833993ce3f4d41d8da16802135e041317b6'
        )

    def test_frame_via_hashlib_b(self) -> None:
        f1 = Frame.from_records([[None, 1], [True, False]])
        with self.assertRaises(TypeError):
            hd = f1.via_hashlib(include_name=False).sha256().hexdigest()

    # ---------------------------------------------------------------------------
    def test_frame_consolidate_a(self) -> None:
        f1 = Frame.from_fields(
            ((10, 20, 30), (40, 20, 30), (False, True, False), (True, False, True)),
            columns=('a', 'b', 'c', 'd'),
            index=('x', 'y', 'z'),
        )
        f2 = f1.consolidate[:]
        self.assertEqual(f2.to_pairs(), f1.to_pairs())
        self.assertEqual(f2._blocks.shapes.tolist(), [(3, 2), (3, 2)])

    def test_frame_consolidate_b1(self) -> None:
        f1 = Frame.from_fields(
            ((10, 20, 30), (40, 20, 30), (2, 4, 5), (5, 6, 8)),
            columns=('a', 'b', 'c', 'd'),
            index=('x', 'y', 'z'),
        )
        f2 = f1.consolidate['a':'c']
        self.assertEqual(f2.to_pairs(), f1.to_pairs())
        self.assertEqual(f2._blocks.shapes.tolist(), [(3, 3), (3,)])

    def test_frame_consolidate_b2(self) -> None:
        f1 = Frame.from_fields(
            ((10, 20, 30), (40, 20, 30), (2, 4, 5), (5, 6, 8)),
            columns=('a', 'b', 'c', 'd'),
            index=('x', 'y', 'z'),
        )
        f2 = f1.consolidate['b':'c']
        self.assertEqual(f2.to_pairs(), f1.to_pairs())
        self.assertEqual(f2._blocks.shapes.tolist(), [(3,), (3, 2), (3,)])

    def test_frame_consolidate_c(self) -> None:
        f1 = Frame.from_fields(
            ((10, 20, 30), (40, 20, 30), (2, 4, 5), (5, 6, 8)),
            columns=('a', 'b', 'c', 'd'),
            index=('x', 'y', 'z'),
            dtypes=np.int64,
        )
        f2 = f1.consolidate['b':'c']
        post = f2.consolidate.status
        self.assertEqual(
            post.to_pairs(),
            (
                ('loc', ((0, 'a'), (1, slice('b', 'c', None)), (2, 'd'))),
                ('iloc', ((0, 0), (1, slice(1, 3, None)), (2, 3))),
                (
                    'dtype',
                    (
                        (0, np.dtype('int64')),
                        (1, np.dtype('int64')),
                        (2, np.dtype('int64')),
                    ),
                ),
                ('shape', ((0, (3,)), (1, (3, 2)), (2, (3,)))),
                ('ndim', ((0, 1), (1, 2), (2, 1))),
                ('owndata', ((0, True), (1, True), (2, True))),
                ('f_contiguous', ((0, True), (1, False), (2, True))),
                ('c_contiguous', ((0, True), (1, True), (2, True))),
            ),
        )

    def test_frame_consolidate_d(self) -> None:
        f1 = Frame.from_fields(
            ((10, 20, 30), (40, 20, 30), (2, 4, 5), (5, 6, 8)),
            columns=('a', 'b', 'c', 'd'),
            index=('x', 'y', 'z'),
            dtypes=np.int64,
        )
        f2 = f1.consolidate['b':'c']
        f3 = f2.consolidate()
        self.assertEqual(
            f3.consolidate.status.to_pairs(),
            (
                ('loc', ((0, slice('a', 'd', None)),)),
                ('iloc', ((0, slice(0, None, None)),)),
                ('dtype', ((0, np.dtype('int64')),)),
                ('shape', ((0, (3, 4)),)),
                ('ndim', ((0, 2),)),
                ('owndata', ((0, True),)),
                ('f_contiguous', ((0, False),)),
                ('c_contiguous', ((0, True),)),
            ),
        )

    def test_frame_consolidate_e(self) -> None:
        f1 = Frame.from_fields(
            (
                (10, 20, 30),
                (40, 20, 30),
                (2, 4, 5),
                (5, 6, 8),
                (1, 0, 1),
            ),
            columns=('a', 'b', 'c', 'd', 'e'),
            index=('x', 'y', 'z'),
            dtypes=np.int64,
        )
        f2 = f1.consolidate[['a', 'c', 'e']]
        self.assertTrue(f1.consolidate.status.equals(f2.consolidate.status))
        f3 = f2.consolidate()
        self.assertEqual(
            f3.consolidate.status.to_pairs(),
            (
                ('loc', ((0, slice('a', 'e', None)),)),
                ('iloc', ((0, slice(0, None, None)),)),
                ('dtype', ((0, np.dtype('int64')),)),
                ('shape', ((0, (3, 5)),)),
                ('ndim', ((0, 2),)),
                ('owndata', ((0, True),)),
                ('f_contiguous', ((0, False),)),
                ('c_contiguous', ((0, True),)),
            ),
        )

    def test_frame_consolidate_f(self) -> None:
        f1 = Frame.from_fields(
            (
                (10, 20, 30),
                (40, 20, 30),
                (2, 4, 5),
                (5, 6, 8),
                (1, 0, 1),
            ),
            columns=('a', 'b', 'c', 'd', 'e'),
            index=('x', 'y', 'z'),
            dtypes=np.int64,
            consolidate_blocks=True,
        )
        # NOTE: this will take a consolidated Frame and break it into three such that the target column is consolidated
        f2 = f1.consolidate['c']
        self.assertEqual(
            f2.consolidate.status['shape'].to_pairs(),
            ((0, (3, 2)), (1, (3,)), (2, (3, 2))),
        )

        f3 = f1.consolidate[:]
        self.assertEqual(
            f3.consolidate.status['shape'].values[0],
            (3, 5),
        )

    # ---------------------------------------------------------------------------
    def test_frame_reduce_a(self) -> None:
        f1 = ff.parse('v(int,bool)|s(4,8)')
        f2 = f1.reduce.from_label_map({1: np.min, 2: np.min}).to_frame()
        self.assertEqual(f2.to_pairs(), ((1, ((None, False),)), (2, ((None, -3648),))))

    # ---------------------------------------------------------------------------
    def test_frame_immutable_a(self) -> None:
        f1 = ff.parse('v(int,bool)|s(3,4)')
        with self.assertRaises(ImmutableTypeError):
            f1['b'] = -1

    def test_frame_immutable_b(self) -> None:
        f1 = ff.parse('v(int,bool)|s(3,4)')
        with self.assertRaises(ImmutableTypeError):
            f1.loc[1] = -1

    def test_frame_immutable_c(self) -> None:
        f1 = ff.parse('v(int,bool)|s(3,4)')
        with self.assertRaises(ImmutableTypeError):
            f1.iloc[1] = -1

    def test_frame_immutable_d(self) -> None:
        f1 = ff.parse('v(int,bool)|s(3,4)')
        with self.assertRaises(ImmutableTypeError):
            f1.iloc[np.array([0, 2])] = -1

    # ---------------------------------------------------------------------------

    def test_frame_selection_exceptions_a(self) -> None:
        f1 = ff.parse('v(int,bool)|s(3,4)|c(I,str)|i(I,str)')
        with self.assertRaises(KeyError):
            _ = f1[-99]
        with self.assertRaises(KeyError):
            _ = f1['z']
        with self.assertRaises(KeyError):
            _ = f1[['z', 'x']]

        with self.assertRaises(IndexError):
            _ = f1[np.array([False, True, True, False, True])]

    def test_frame_selection_exceptions_b(self) -> None:
        f1 = ff.parse('v(int,bool)|s(3,4)|c(I,str)|i(I,str)')
        with self.assertRaises(KeyError):
            _ = f1.loc[-99]
        with self.assertRaises(KeyError):
            _ = f1.loc['z']
        with self.assertRaises(KeyError):
            _ = f1.loc['z', 'x']

        with self.assertRaises(IndexError):
            _ = f1.loc[np.array([False, True, True, False, True])]

    def test_frame_selection_exceptions_c(self) -> None:
        f1 = ff.parse('v(int,bool)|s(3,4)|c(I,str)|i(I,str)')
        with self.assertRaises(IndexError):
            _ = f1.iloc[-99]
        with self.assertRaises(IndexError):
            _ = f1.iloc[0, -99]
        with self.assertRaises(IndexError):
            _ = f1.iloc[np.array([False, True, True, False, True])]

    def test_frame_selection_exceptions_d(self) -> None:
        f1 = ff.parse('s(4,5)|i(IH,(str,str))|c(IH,(int,int))')
        with self.assertRaises(KeyError):
            _ = f1.loc[-99]
        with self.assertRaises(KeyError):
            _ = f1.loc['z']
        with self.assertRaises(KeyError):
            _ = f1.loc['z', 'x']
        with self.assertRaises(KeyError):
            _ = f1.loc[['z', 'q'], 'x']

        with self.assertRaises(IndexError):
            _ = f1.loc[np.array([False, True, True, False, True])]

    def test_frame_selection_exceptions_e(self) -> None:
        f1 = ff.parse('s(4,5)|i(IH,(str,str))|c(IH,(int,int))')
        with self.assertRaises(IndexError):
            _ = f1.iloc[-99]
        with self.assertRaises(IndexError):
            _ = f1.iloc[-99, -99]
        with self.assertRaises(IndexError):
            _ = f1.iloc[[-99, -1000], 3]

        with self.assertRaises(IndexError):
            _ = f1.loc[np.array([False, True, True, False, True])]


if __name__ == '__main__':
    unittest.main()
