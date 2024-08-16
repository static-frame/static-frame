from __future__ import annotations

import frame_fixtures as ff
import numpy as np
import typing_extensions as tp

import static_frame as sf
from static_frame import Frame
from static_frame import FrameGO
from static_frame import HLoc
from static_frame import IndexDate
from static_frame import IndexHierarchy
from static_frame import IndexYear
from static_frame import Series
from static_frame import TypeBlocks
from static_frame.core.exception import AxisInvalid
from static_frame.core.util import TLabel
from static_frame.test.test_case import TestCase

nan = np.nan


class TestUnit(TestCase):

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
        self.assertEqual([x for x in f1], ['p', 'q', 'r', 's', 't']) # type: ignore

    def test_frame_iter_array_a(self) -> None:

        records = (
                (1, 2, 'a', False, True),
                (30, 50, 'b', True, False))

        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('x','y'))

        self.assertEqual(
                next(iter(f1.iter_array(axis=0))).tolist(), # type: ignore
                [1, 30])

        self.assertEqual(
                next(iter(f1.iter_array(axis=1))).tolist(), # type: ignore
                [1, 2, 'a', False, True])

    def test_frame_iter_array_b(self) -> None:

        arrays = list(np.random.rand(1000) for _ in range(100))
        f1 = Frame.from_items(
                zip(range(100), arrays)
                )

        # iter columns
        post = f1.iter_array(axis=0).apply_pool(np.sum, max_workers=4, use_threads=True)
        self.assertEqual(post.shape, (100,))
        self.assertAlmostEqual(f1.sum().sum(), post.sum())

        post = f1.iter_array(axis=0).apply_pool(np.sum, max_workers=4, use_threads=False)
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
        self.assertTrue(post.dtype == float) # type: ignore

    def test_frame_iter_array_f(self) -> None:

        f = sf.Frame(np.arange(12).reshape(3,4),
                index=IndexDate.from_date_range('2020-01-01', '2020-01-03'))

        post = f.iter_array(axis=0).apply(np.sum, name='foo')
        self.assertEqual(post.name, 'foo')

        self.assertEqual(
                f.iter_array(axis=0).apply(np.sum).to_pairs(),
                ((0, 12), (1, 15), (2, 18), (3, 21))
                )

        self.assertEqual(
                f.iter_array(axis=1).apply(np.sum).to_pairs(),
                ((np.datetime64('2020-01-01'), 6), (np.datetime64('2020-01-02'), 22), (np.datetime64('2020-01-03'), 38))
                )

    def test_frame_iter_array_g(self) -> None:

        f = sf.FrameGO(index=IndexDate.from_date_range('2020-01-01', '2020-01-03'))
        post = list(f.iter_array(axis=0))
        self.assertEqual(post, [])

        post = list(f.iter_array(axis=1))
        self.assertEqual([x.tolist() for x in post], [[], [], []])

    #---------------------------------------------------------------------------

    def test_frame_iter_tuple_a(self) -> None:
        post = tuple(sf.Frame.from_elements(range(5)).iter_tuple(axis=0, constructor=tuple))
        self.assertEqual(post, ((0, 1, 2, 3, 4),))

    def test_frame_iter_tuple_b(self) -> None:
        post = tuple(sf.Frame.from_elements(range(3), index=tuple('abc')).iter_tuple(axis=0))
        self.assertEqual(post, ((0, 1, 2),))

        self.assertEqual(tuple(post[0]._asdict().items()),
                (('a', 0), ('b', 1), ('c', 2))
                )

    def test_frame_iter_tuple_c(self) -> None:
        with self.assertRaises(AxisInvalid):
            post = tuple(sf.Frame.from_elements(range(5)).iter_tuple(axis=2))

    def test_frame_iter_tuple_d(self) -> None:
        f = sf.FrameGO(index=IndexDate.from_date_range('2020-01-01', '2020-01-03'))
        post = list(f.iter_tuple(constructor=tuple, axis=0))
        self.assertEqual(post, [])

        post = list(f.iter_tuple(axis=1))
        self.assertEqual([len(x) for x in post], [0, 0, 0])

    def test_frame_iter_tuple_e(self) -> None:
        records = (
                (1, 2, 'a', False, True),
                (30, 50, 'b', True, False))

        f1 = FrameGO.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('x','y'))

        class Record(tp.NamedTuple):
            x: object
            y: object

        post1 = list(f1.iter_tuple(constructor=Record))
        self.assertTrue(all(isinstance(x, Record) for x in post1))

        post2 = list(f1.iter_tuple(constructor=tuple))
        self.assertEqual(post2,
                [(1, 30), (2, 50), ('a', 'b'), (False, True), (True, False)])

    def test_frame_iter_tuple_f(self) -> None:
        f = ff.parse('s(3,2)|v(dtD,dtY)')
        post = tuple(f.iter_tuple(constructor=tuple, axis=1))
        self.assertEqual(len(post[0]), 2)
        self.assertEqual(post[0][1].dtype, f.iloc[0, 1].dtype)
        self.assertEqual(post[0],
            (np.datetime64('2210-12-26'), np.datetime64('164167'))
            )

    def test_frame_iter_tuple_g(self) -> None:
        # NOTE: this test demonstrate the utility of mapping functions on the only iterable axis type (tuple, ignoring SeriesHE) that is hashable
        columns = tuple('pqrs')
        index = tuple('zxwy')
        records = (('A', 1,  False, False),
                   ('A', 2,  True, False),
                   ('B', 1,  False, False),
                   ('B', 2,  True, True))
        f = Frame.from_records(records, columns=columns, index=index)
        post = f[['r', 's']].iter_tuple(axis=1, constructor=tuple).map_any({(False, False): None}) # type: ignore
        self.assertEqual(post.to_pairs(),
                (('z', None), ('x', (True, False)), ('w', None), ('y', (True, True)))
                )

    def test_frame_iter_tuple_h(self) -> None:
        # NOTE: this test demonstrate the utility of mapping functions on the only iterable axis type (tuple, ignoring SeriesHE) that is hashable
        columns = tuple('pqrs')
        index = tuple('zxwy')
        records = (('A', 1,  False, False),
                   ('A', 2,  True, False),
                   ('B', 1,  False, False),
                   ('B', 2,  True, True))
        f = Frame.from_records(records, columns=columns, index=index)
        post = f[['r', 's']].iter_tuple_items(axis=1, constructor=tuple).map_fill ({('w', (False, False)): None}, fill_value=0) # type: ignore
        self.assertEqual(post.to_pairs(),
            (('z', 0), ('x', 0), ('w', None), ('y', 0))
            )

    def test_frame_iter_tuple_i1(self) -> None:
        from dataclasses import dataclass

        records = (
                (1,  False, True),
                (30, True, False))

        f1 = FrameGO.from_records(records,
                columns=('p', 'q', 'r'),
                index=('x','y'))

        @dataclass
        class Record:
            p: int
            q: bool
            r: bool

        post1 = list(f1.iter_tuple(axis=1, constructor=Record))
        self.assertTrue(all(isinstance(x, Record) for x in post1))
        self.assertEqual([x.p for x in post1], [1, 30])
        self.assertEqual([x.r for x in post1], [True, False])

    def test_frame_iter_tuple_i2(self) -> None:
        from dataclasses import dataclass

        records = (
                (1,  False, True),
                (30, True, False))

        f1 = FrameGO.from_records(records,
                columns=('p', 'q', 'r'),
                index=('x','y'))

        @dataclass
        class Record:
            p: int
            q: bool
            r: bool

        ctor = lambda args: Record(**dict(zip(('p', 'q', 'r'), args)))
        post1 = list(f1.iter_tuple(axis=1, constructor=ctor))
        self.assertTrue(all(isinstance(x, Record) for x in post1))
        self.assertEqual([x.p for x in post1], [1, 30])
        self.assertEqual([x.r for x in post1], [True, False])




    #---------------------------------------------------------------------------

    def test_frame_iter_series_a(self) -> None:
        f1 = ff.parse('f(Fg)|s(2,8)|i(I,str)|c(Ig,str)|v(int)')
        post1 = tuple(f1.iter_series(axis=0))
        self.assertEqual(len(post1), 8)
        self.assertEqual(post1[0].to_pairs(),
                (('zZbu', -88017), ('ztsv', 92867)))

        post2 = tuple(f1.iter_series(axis=1))
        self.assertEqual(len(post2), 2)
        self.assertEqual(post2[0].to_pairs(),
                (('zZbu', -88017), ('ztsv', 162197), ('zUvW', -3648), ('zkuW', 129017), ('zmVj', 58768), ('z2Oo', 84967), ('z5l6', 146284), ('zCE3', 137759)))

    def test_frame_iter_series_b(self) -> None:
        f1 = ff.parse('s(10,4)|i(I,str)|c(I,str)|v(float)')

        post1 = f1.iter_series(axis=1).apply(lambda s: s.sum(), dtype=float)
        self.assertEqual(post1.dtype, float)
        self.assertEqual(post1.shape, (10,))

        post2 = f1[sf.ILoc[0]].iter_element().apply(lambda e: e == 647.9, dtype=bool)
        self.assertEqual(post2.dtype, bool)
        self.assertEqual(post2.shape, (10,))
        self.assertEqual(post2.sum(), 1)

        post3 = f1.iter_series(axis=1).apply(lambda s: str(s.sum()), dtype=str)
        self.assertEqual(post3.dtype, np.dtype('<U18'))
        self.assertEqual(post3.shape, (10,))

        post4 = f1.iter_series(axis=1).apply(lambda s: int(s.sum()), dtype=int)
        self.assertEqual(post4.dtype, np.dtype(int))
        self.assertEqual(post4.shape, (10,))

        post5 = f1.iter_series(axis=1).apply(lambda s: int(s.sum()), dtype=object)
        self.assertEqual(post5.dtype, object)
        self.assertEqual(post5.shape, (10,))

    #---------------------------------------------------------------------------
    def test_frame_iter_series_items_a(self) -> None:
        f1 = ff.parse('f(Fg)|s(2,8)|i(I,str)|c(Ig,str)|v(int)')
        post1 = tuple(f1.iter_series_items(axis=0))
        self.assertEqual([(k, v.values.tolist()) for (k, v) in post1],
                [('zZbu', [-88017, 92867]), ('ztsv', [162197, -41157]), ('zUvW', [-3648, 91301]), ('zkuW', [129017, 35021]), ('zmVj', [58768, 146284]), ('z2Oo', [84967, 13448]), ('z5l6', [146284, 170440]), ('zCE3', [137759, -62964])])

        post2 = tuple(f1.iter_series_items(axis=1))
        self.assertEqual([(k, v.values.tolist()) for (k, v) in post2],
                [('zZbu', [-88017, 162197, -3648, 129017, 58768, 84967, 146284, 137759]), ('ztsv', [92867, -41157, 91301, 35021, 146284, 13448, 170440, -62964])]
                )

    #---------------------------------------------------------------------------

    def test_frame_iter_tuple_items_a(self) -> None:
        records = (
                (1, 2, 'a', False, True),
                (30, 50, 'b', True, False))

        f1 = FrameGO.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('x','y'))

        post1 = list(f1.iter_tuple_items(constructor=list))
        self.assertEqual(post1, [('p', [1, 30]), ('q', [2, 50]), ('r', ['a', 'b']), ('s', [False, True]), ('t', [True, False])])

    #---------------------------------------------------------------------------

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

        self.assertEqual(list(f1.iter_element(axis=1)),
                [2, 30, 2, 30, 2, 34, 95, 73, 'a', 'b', 'c', 'd', False, True, False, True, False, False, False, True])

        self.assertEqual([x for x in f1.iter_element_items()],
                [(('w', 'p'), 2), (('w', 'q'), 2), (('w', 'r'), 'a'), (('w', 's'), False), (('w', 't'), False), (('x', 'p'), 30), (('x', 'q'), 34), (('x', 'r'), 'b'), (('x', 's'), True), (('x', 't'), False), (('y', 'p'), 2), (('y', 'q'), 95), (('y', 'r'), 'c'), (('y', 's'), False), (('y', 't'), False), (('z', 'p'), 30), (('z', 'q'), 73), (('z', 'r'), 'd'), (('z', 's'), True), (('z', 't'), True)])


        post1 = f1.iter_element().apply(lambda x: '_' + str(x) + '_')

        self.assertEqual(post1.to_pairs(0),
                (('p', (('w', '_2_'), ('x', '_30_'), ('y', '_2_'), ('z', '_30_'))), ('q', (('w', '_2_'), ('x', '_34_'), ('y', '_95_'), ('z', '_73_'))), ('r', (('w', '_a_'), ('x', '_b_'), ('y', '_c_'), ('z', '_d_'))), ('s', (('w', '_False_'), ('x', '_True_'), ('y', '_False_'), ('z', '_True_'))), ('t', (('w', '_False_'), ('x', '_False_'), ('y', '_False_'), ('z', '_True_')))))

        post2 = f1.iter_element(axis=1).apply(lambda x: '_' + str(x) + '_')

        self.assertEqual(post2.to_pairs(0),
                (('p', (('w', '_2_'), ('x', '_30_'), ('y', '_2_'), ('z', '_30_'))), ('q', (('w', '_2_'), ('x', '_34_'), ('y', '_95_'), ('z', '_73_'))), ('r', (('w', '_a_'), ('x', '_b_'), ('y', '_c_'), ('z', '_d_'))), ('s', (('w', '_False_'), ('x', '_True_'), ('y', '_False_'), ('z', '_True_'))), ('t', (('w', '_False_'), ('x', '_False_'), ('y', '_False_'), ('z', '_True_')))))


        post3 = f1.iter_element(axis=1).apply(lambda x: '_' + str(x) + '_', dtype=str)

        self.assertEqual(post3.to_pairs(0),
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
        post = f1.iter_element().map_any({2: 200, False: 200})

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

    def test_frame_iter_element_d(self) -> None:
        f1 = sf.Frame.from_elements(['I', 'II', 'III'], columns=('A',))
        f2 = sf.Frame.from_elements([67, 28, 99], columns=('B',), index=('I', 'II', 'IV'))

        post = f1['A'].iter_element().map_any(f2['B'])

        # same index is given to returned Series
        self.assertEqual(id(f1.index), id(post.index))
        # if we do not match the mapping, we keep the value.
        self.assertEqual(post.to_pairs(),
                ((0, 67), (1, 28), (2, 'III')))

    def test_frame_iter_element_e(self) -> None:
        f1 = Frame.from_records(np.arange(9).reshape(3, 3))

        self.assertEqual(list(f1.iter_element(axis=1)),
                [0, 3, 6, 1, 4, 7, 2, 5, 8])

        mapping = {x: x*3 for x in range(9)}
        f2 = f1.iter_element(axis=1).map_all(mapping)
        self.assertEqual([d.kind for d in f2.dtypes.values],
                ['i', 'i', 'i'])

    def test_frame_iter_element_f(self) -> None:
        f1 = Frame.from_records(np.arange(9).reshape(3, 3))

        mapping = {x: x*3 for x in range(3)}
        f2 = f1.iter_element(axis=1).map_fill(mapping, fill_value=-1)
        self.assertEqual(f2.to_pairs(),
            ((0, ((0, 0), (1, -1), (2, -1))), (1, ((0, 3), (1, -1), (2, -1))), (2, ((0, 6), (1, -1), (2, -1))))
            )

    def test_frame_iter_element_g1(self) -> None:
        f1 = Frame(np.arange(9).reshape(3, 3)).rename(index='a', columns='b')
        f2 = f1.iter_element().apply(str)
        self.assertEqual(f1.columns.name, f2.columns.name)
        self.assertEqual(f1.index.name, f2.index.name)

    def test_frame_iter_element_g2(self) -> None:
        f1 = Frame(np.arange(9).reshape(3, 3), columns=('2021', '1943', '1523')).rename(index='a', columns='b')
        f2 = f1.iter_element().apply(str, columns_constructor=IndexYear)
        self.assertIs(f2.columns.__class__, IndexYear)
        self.assertIs(f2.columns.name, 'b')
        self.assertEqual(f1.shape, f2.shape)

    def test_frame_iter_element_g3(self) -> None:
        f1 = Frame(np.arange(9).reshape(3, 3), index=('2021', '1943', '1523')).rename(index='a', columns='b')
        f2 = f1.iter_element().apply(str, index_constructor=IndexYear)
        self.assertIs(f2.index.__class__, IndexYear)
        self.assertIs(f2.index.name, 'a')
        self.assertEqual(f1.shape, f2.shape)


    #---------------------------------------------------------------------------

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

        with self.assertRaises(AxisInvalid):
            _ = f.iter_group('s', axis=-1).apply(lambda x: x.shape)

        post = f.iter_group('s').apply(lambda x: x.shape)
        self.assertEqual(post.to_pairs(),
                ((False, (2, 3)), (True, (2, 3)))
                )
        self.assertEqual(post.index.name, 's')

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

    def test_frame_iter_group_c(self) -> None:
        columns = tuple('pqrst')
        index = tuple('zxwy')
        records = (('A', 1, 'a', False, False),
                   ('A', 2, 'b', True, False),
                   ('B', 1, 'c', False, False),
                   ('B', 2, 'd', True, True))

        f = Frame.from_records(
                records, columns=columns, index=index, name='foo')

        with self.assertRaises(TypeError):
            next(iter(f.iter_group(foo='x'))) # type: ignore

        with self.assertRaises(TypeError):
            next(iter(f.iter_group(3, 5)))

        self.assertEqual(next(iter(f.iter_group('q'))).to_pairs(0), # type: ignore
                (('p', (('z', 'A'), ('w', 'B'))), ('q', (('z', 1), ('w', 1))), ('r', (('z', 'a'), ('w', 'c'))), ('s', (('z', False), ('w', False))), ('t', (('z', False), ('w', False))))
                )

    def test_frame_iter_group_d(self) -> None:
        f = sf.Frame.from_element(1, columns=[1,2,3], index=['a'])
        empty = f.reindex([])
        self.assertEqual(list(empty.iter_element()), [])
        self.assertEqual(list(empty.iter_group(key=1)), [])

    def test_frame_iter_group_e(self) -> None:
        f = sf.Frame.from_element(None, columns=[1,2,3], index=['a'])
        empty = f.reindex([])
        self.assertEqual(list(empty.iter_element()), [])
        self.assertEqual(list(empty.iter_group(key=1)), [])

    def test_frame_iter_group_f(self) -> None:
        f = sf.Frame(np.arange(3).reshape(1,3), columns=tuple('abc'))
        f = f.drop.loc[0]
        post1 = tuple(f.iter_group(['b','c']))
        self.assertEqual(post1, ())

        post2 = tuple(f.iter_group('a'))
        self.assertEqual(post2, ())

    def test_frame_iter_group_g(self) -> None:
        f = sf.Frame.from_records(
                [['1998-10-12', 1], ['1998-10-13', 2]],
                columns=['A', 'B'],
                dtypes=['datetime64[D]', int]
                )
        post = f.iter_group('A').apply(lambda v: v['B'].sum(),
                index_constructor=IndexDate)

        self.assertEqual(post.index.__class__, IndexDate)
        self.assertEqual(post.to_pairs(),
            ((np.datetime64('1998-10-12'), 1),
            (np.datetime64('1998-10-13'), 2)))

    def test_frame_iter_group_h(self) -> None:
        f1 = sf.Frame(np.arange(25).reshape(5, 5), index=tuple("ABCDE"))

        with self.assertRaises(KeyError):
            f1.iter_group(-99).apply(lambda x: x)

        with self.assertRaises(KeyError):
            s2 = f1.iter_group('z').apply(lambda x: x)



    #---------------------------------------------------------------------------
    def test_frame_iter_group_array_a(self) -> None:
        f1 = ff.parse('s(7,3)|v(int)').assign[1].apply(
                lambda s: s % 3)

        post = tuple(f1.iter_group_array(1))
        self.assertEqual([p.shape for p in post],
                [(3, 3), (4, 3)]
                )
        self.assertEqual([p.__class__ for p in post],
                [np.ndarray, np.ndarray]
                )

    def test_frame_iter_group_array_b(self) -> None:
        f1 = ff.parse('s(7,3)|v(int)').assign[1].apply(
                lambda s: s % 3)

        post = tuple(f1.iter_group_array(1, drop=True))
        self.assertEqual([p.shape for p in post],
                [(3, 2), (4, 2)]
                )
        self.assertEqual([p.__class__ for p in post],
                [np.ndarray, np.ndarray]
                )

    #---------------------------------------------------------------------------
    def test_frame_iter_group_array_items_a(self) -> None:
        f1 = ff.parse('s(7,3)|v(int)').assign[1].apply(
                lambda s: s % 3)

        post = tuple(f1.iter_group_array_items(1))
        self.assertEqual([p[1].shape for p in post],
                [(3, 3), (4, 3)]
                )
        self.assertEqual([p[1].__class__ for p in post],
                [np.ndarray, np.ndarray]
                )
        self.assertEqual([p[0] for p in post],
                [0, 2]
                )

    #---------------------------------------------------------------------------

    def test_frame_iter_group_items_a(self) -> None:
        # testing a hierarchical index and columns, selecting column with a tuple
        records = (
                ('a', 999999, 0.1),
                ('a', 201810, 0.1),
                ('b', 999999, 0.4),
                ('b', 201810, 0.4))
        f1 = Frame.from_records(records, columns=list('abc'))

        f1 = f1.set_index_hierarchy(['a', 'b'], drop=False)
        f1 = f1.relabel_level_add(columns='i')

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

    def test_frame_iter_group_items_c1(self) -> None:
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


    def test_frame_iter_group_items_c2(self) -> None:
        # Test optimized sorting approach. Data must have a non-object dtype and key must be single
        data = np.array([[0, 1, 1, 3],
                         [3, 3, 2, 3],
                         [5, 5, 1, 3],
                         [7, 2, 2, 4]])

        frame = sf.Frame(data, columns=tuple('abcd'), index=tuple('wxyz'))
        groups = list(frame.iter_group_items('w', axis=1))
        expected_pairs = [
                (('a', (('w', 0), ('x', 3), ('y', 5), ('z', 7))),),
                (('b', (('w', 1), ('x', 3), ('y', 5), ('z', 2))),
                 ('c', (('w', 1), ('x', 2), ('y', 1), ('z', 2)))),
                (('d', (('w', 3), ('x', 3), ('y', 3), ('z', 4))),)]

        self.assertEqual([0, 1, 3], [group[0] for group in groups])
        self.assertEqual(expected_pairs, [group[1].to_pairs(axis=0) for group in groups])


    def test_frame_iter_group_items_d1(self) -> None:
        # Test iterating with multiple key selection
        data = np.array([[0, 1, 1, 3],
                         [3, 3, 2, 3],
                         [5, 5, 1, 3],
                         [7, 2, 2, 4]])

        frame = sf.Frame(data, columns=tuple('abcd'), index=tuple('wxyz'))

        # Column
        groups = list(frame.iter_group_items(['c', 'd'], axis=0))
        self.assertEqual([(1, 3), (2, 3), (2, 4)], [group[0] for group in groups])
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

        self.assertEqual(expected_pairs, [group[1].to_pairs(axis=0) for group in groups])


    def test_frame_iter_group_items_d2(self) -> None:
        # Test iterating with multiple key selection
        data = np.array([[0, 1, 1, 3],
                         [3, 3, 2, 3],
                         [5, 5, 1, 3],
                         [7, 2, 2, 4]])

        frame = sf.Frame(data, columns=tuple('abcd'), index=tuple('wxyz'))
        # Index
        groups = list(frame.iter_group_items(['x', 'y'], axis=1))
        self.assertEqual([(2, 1), (3, 3), (3, 5)], [group[0] for group in groups])

        expected_pairs = [
                (('c', (('w', 1), ('x', 2), ('y', 1), ('z', 2))),),
                (('d', (('w', 3), ('x', 3), ('y', 3), ('z', 4))),),
                (('a', (('w', 0), ('x', 3), ('y', 5), ('z', 7))),
                 ('b', (('w', 1), ('x', 3), ('y', 5), ('z', 2)))),
        ]

        self.assertEqual(expected_pairs, [group[1].to_pairs(axis=0) for group in groups])

    def test_frame_iter_group_items_e(self) -> None:
        columns = tuple('pqrst')
        index = tuple('zxwy')
        records = (('A', 1, 'a', False, False),
                   ('A', 2, 'b', True, False),
                   ('B', 1, 'c', False, False),
                   ('B', 2, 'd', True, True))

        f = Frame.from_records(
                records, columns=columns, index=index,name='foo')

        #  using an array to select
        self.assertEqual(
                tuple(k for k, v in f.iter_group_items(f.columns == 's')),
                ((False,), (True,))
                )

        self.assertEqual(
                tuple(k for k, v in f.iter_group_items(f.columns.isin(('p', 't')))),
                (('A', False), ('B', False), ('B', True))
                )
        self.assertEqual(
                tuple(k for k, v in f.iter_group_items(['s', 't'])),
                ((False, False), (True, False), (True, True))
                )

        self.assertEqual(
                tuple(k for k, v in f.iter_group_items(slice('s','t'))),
                ((False, False), (True, False), (True, True))
                )

    def test_frame_iter_group_items_f(self) -> None:

        objs = [object() for _ in range(2)]
        data = [[1, 2, objs[0]], [3, 4, objs[0]], [5, 6, objs[1]]]
        f = sf.Frame.from_records(data, columns=tuple('abc'))

        post1 = {k: v for k, v in f.iter_group_items('c')}
        post2 = {k[0]: v for k, v in f.iter_group_items(['c'])} # as a list, this gets a multiple key

        self.assertEqual(len(post1), 2)
        self.assertEqual(len(post1), len(post2))

        obj_a = objs[0]
        obj_b = objs[1]

        self.assertEqual(post1[obj_a].shape, (2, 3))
        self.assertEqual(post1[obj_a].shape, post2[obj_a].shape)
        self.assertEqual(post1[obj_a].to_pairs(0),
                (('a', ((0, 1), (1, 3))), ('b', ((0, 2), (1, 4))), ('c', ((0, obj_a), (1, obj_a)))))
        self.assertEqual(post2[obj_a].to_pairs(0),
                (('a', ((0, 1), (1, 3))), ('b', ((0, 2), (1, 4))), ('c', ((0, obj_a), (1, obj_a)))))


        self.assertEqual(post1[obj_b].shape, (1, 3))
        self.assertEqual(post1[obj_b].shape, post2[obj_b].shape)
        self.assertEqual(post1[obj_b].to_pairs(0),
                (('a', ((2, 5),)), ('b', ((2, 6),)), ('c', ((2, obj_b),))))
        self.assertEqual(post2[obj_b].to_pairs(0),
                (('a', ((2, 5),)), ('b', ((2, 6),)), ('c', ((2, obj_b),))))

    #---------------------------------------------------------------------------

    def test_frame_iter_group_labels_a(self) -> None:

        records = (
                (2, 2, 'a', False, False),
                (30, 34, 'b', True, False),
                (2, 95, 'c', False, False),
                )
        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('x', 'y', 'z'))

        with self.assertRaises(TypeError):
            f1.iter_group_labels(3, 4)

        with self.assertRaises(TypeError):
            f1.iter_group_labels(foo=4) # type: ignore

        post = tuple(f1.iter_group_labels(0, axis=0))

        self.assertEqual(len(post), 3)
        self.assertEqual(
                f1.iter_group_labels(0, axis=0).apply(lambda x: x[['p', 'q']].values.sum()).to_pairs(),
                (('x', 4), ('y', 64), ('z', 97))
                )

    def test_frame_iter_group_labels_b(self) -> None:

        records = (
                (2, 2, 'a', 'q', False, False),
                (30, 34, 'b', 'c', True, False),
                (2, 95, 'c', 'd', False, False),
                )
        f1 = Frame.from_records(records,
                columns=IndexHierarchy.from_product((1, 2, 3), ('a', 'b')),
                index=('x', 'y', 'z'))

        # with axis 1, we are grouping based on columns while maintain the index
        post_tuple = tuple(f1.iter_group_labels(1, axis=1))
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
                f1.iter_group_labels(1, axis=1).apply(lambda x: x.iloc[:, 0].sum()).to_pairs(),
                (('a', 34), ('b', 131))
                )

    def test_frame_iter_group_labels_c(self) -> None:
        columns = tuple('pqrst')
        index = tuple('zxwy')
        records = (('A', 1, 'a', False, False),
                   ('A', 2, 'b', True, False),
                   ('B', 1, 'c', False, False),
                   ('B', 2, 'd', True, True))

        f = Frame.from_records(
                records, columns=columns, index=index,name='foo')
        f = f.set_index_hierarchy(('p', 'q'), drop=True)

        with self.assertRaises(AxisInvalid):
            _ = f.iter_group_labels_items(0, axis=-1).apply(lambda k, x: f'{k}:{x.size}')

        post = f.iter_group_labels_items(0).apply(lambda k, x: f'{k}:{x.size}')

        self.assertEqual(post.to_pairs(),
                (('A', 'A:6'), ('B', 'B:6'))
        )

    def test_frame_iter_group_labels_d(self) -> None:
        columns = tuple('pqrst')
        index = tuple('zxwy')
        records = (('A', 1, 'a', False, 4),
                   ('A', 2, 'b', True, 3),
                   ('B', 1, 'c', False, 2),
                   ('B', 2, 'd', True, 1))

        f = Frame.from_records(
                records, columns=columns, index=index,name='foo')
        f = f.set_index_hierarchy(('p', 'q', 'r'), drop=True)

        post = tuple(f.iter_group_labels_items((1, 2), axis=0))
        self.assertEqual([p[0] for p in post],
                [(1, 'a'), (1, 'c'), (2, 'b'), (2, 'd')]
                )

        self.assertEqual([p[1].values.tolist() for p in post],
                [[[False, 4]], [[False, 2]], [[True, 3]], [[True, 1]]]
                )

    def test_frame_iter_group_labels_e(self) -> None:
        index = tuple('pq')
        columns = IndexHierarchy._from_type_blocks(TypeBlocks.from_blocks((
                np.array(('A', 'A', 'B', 'B', 'B')),
                np.array((4, 2, 1, 0, 4)),
                np.array(('b', 'a', 'c', 'a', 'b')),
                )))

        records = (
                   (True, False, 1, 2, 1),
                   (False, True, 30, 8,7),
                   )

        f = Frame.from_records(
                records, columns=columns, index=index)
        post = tuple(f.iter_group_labels_items((2, 1), axis=1))
        self.assertEqual([p[0] for p in post],
                [('a', 0), ('a', 2), ('b', 4), ('c', 1)]
                )
        self.assertEqual([p[1].values.tolist() for p in post],
                [[[2], [8]], [[False], [True]], [[True, 1], [False, 7]], [[1], [30]]]
                )

    def test_frame_iter_group_labels_f(self) -> None:
        columns = tuple('pqrst')
        index = tuple('zxwy')
        records = (('A', 1, 'a', False, 4),
                   (None, 2, 'b', True, 3),
                   ('B', 1, 'c', False, 2),
                   ('B', None, 'd', True, 1))

        f = Frame.from_records(
                records, columns=columns, index=index,name='foo')
        f = f.set_index_hierarchy(('p', 'q'), drop=True)

        post = tuple(f.iter_group_labels_items(0, axis=0))
        self.assertEqual([p[0] for p in post],
                ['A', None, 'B']
                )
        self.assertEqual([p[1].values.tolist() for p in post],
            [[['a', False, 4]], [['b', True, 3]], [['c', False, 2], ['d', True, 1]]]
            )

    #---------------------------------------------------------------------------

    def test_frame_iter_group_labels_array_a(self) -> None:
        columns = tuple('pqrs')
        index = tuple('zxwy')
        records = (('A', 1,  False, False),
                   ('A', 2,  True, False),
                   ('B', 1,  False, False),
                   ('B', 2,  True, True))

        f = Frame.from_records(records, columns=columns, index=index)
        f = f.set_index_hierarchy(('p', 'q'), drop=True)

        post1 = tuple(f.iter_group_labels_array(0))
        self.assertEqual(len(post1), 2)
        self.assertEqual([a.__class__ for a in post1], [np.ndarray, np.ndarray])
        self.assertEqual([a.shape for a in post1], [(2, 2), (2, 2)])

    #---------------------------------------------------------------------------

    def test_frame_iter_group_labels_array_items_a(self) -> None:
        columns = tuple('pqrs')
        index = tuple('zxwy')
        records = (('A', 1,  False, False),
                   ('A', 2,  True, False),
                   ('B', 1,  False, False),
                   ('B', 2,  True, True))

        f = Frame.from_records(records, columns=columns, index=index)
        f = f.set_index_hierarchy(('p', 'q'), drop=True)

        post1 = tuple(f.iter_group_labels_array_items(0))
        self.assertEqual(len(post1), 2)
        self.assertEqual([a[1].__class__ for a in post1], [np.ndarray, np.ndarray])
        self.assertEqual([a[1].shape for a in post1], [(2, 2), (2, 2)])
        self.assertEqual([a[0] for a in post1], ['A', 'B'])

    def test_frame_iter_group_labels_array_items_b(self) -> None:
        f = sf.Frame(np.arange(8).reshape(2, 4),
                columns=sf.IndexHierarchy.from_labels(
                        ((1, 'a'), (2, 'b'), (1, 'b'), (2, 'a'))
                ))
        post1 = tuple(f.iter_group_labels_array_items(1, axis=1))
        self.assertEqual(len(post1), 2)
        self.assertEqual([a[1].__class__ for a in post1], [np.ndarray, np.ndarray])
        self.assertEqual([a[1].shape for a in post1], [(2, 2), (2, 2)])
        self.assertEqual([a[0] for a in post1], ['a', 'b'])

    #---------------------------------------------------------------------------

    def test_frame_iter_group_other_a(self) -> None:
        columns = tuple('pqr')
        index = tuple('zxwy')
        records = (('A', 1, False),
                   ('A', 2, True),
                   ('B', 1, False),
                   ('B', 2, True))
        f = Frame.from_records(records, columns=columns, index=index)
        # asxis 0 means for other grouping means that they key is a column key
        post1, post2 = list(f.iter_group_other(axis=0, other=(0, 0, 0, 1)))
        self.assertEqual(post1.to_pairs(),
                (('p', (('z', 'A'), ('x', 'A'), ('w', 'B'))), ('q', (('z', 1), ('x', 2), ('w', 1))), ('r', (('z', False), ('x', True), ('w', False))))
                )
        self.assertEqual(post2.to_pairs(),
                (('p', (('y', 'B'),)), ('q', (('y', 2),)), ('r', (('y', True),)))
                )

        post3, post4 = list(f.iter_group_other(axis=1, other=(1, 0, 1)))
        self.assertEqual(post3.to_pairs(),
                (('q', (('z', 1), ('x', 2), ('w', 1), ('y', 2))),)
                )
        self.assertEqual(post4.to_pairs(),
                (('p', (('z', 'A'), ('x', 'A'), ('w', 'B'), ('y', 'B'))), ('r', (('z', False), ('x', True), ('w', False), ('y', True))))
                )

    def test_frame_iter_group_other_b(self) -> None:
        columns = tuple('pqr')
        index = tuple('zxwy')
        records = (('A', 1, False),
                   ('A', 2, True),
                   ('B', 1, False),
                   ('B', 2, True))
        f = Frame.from_records(records, columns=columns, index=index)
        # axis 0 means for other grouping means that they key is a column key
        post1, post2 = list(f.iter_group_other(axis=0, other=f[['q', 'r']]))
        self.assertEqual(post1.to_pairs(),
                (('p', (('z', 'A'), ('w', 'B'))), ('q', (('z', 1), ('w', 1))), ('r', (('z', False), ('w', False))))
                )
        self.assertEqual(post2.to_pairs(),
                (('p', (('x', 'A'), ('y', 'B'))), ('q', (('x', 2), ('y', 2))), ('r', (('x', True), ('y', True))))
                )

    def test_frame_iter_group_other_c(self) -> None:
        columns = tuple('pqr')
        index = tuple('zxwy')
        records = ((1, 1, False),
                   (1, 1, False),
                   (5, 1, False),
                   (8, 2, True))
        f = Frame.from_records(records, columns=columns, index=index)
        # axis 0 means for other grouping means that they key is a column key
        (g1, post1), (g2, post2) = list(f.iter_group_other_items(axis=1,
                other=f.iloc[:2, :2],
                fill_value = None
                ))
        self.assertEqual(g1, (1, 1))
        self.assertEqual(post1.to_pairs(),
                (('p', (('z', 1), ('x', 1), ('w', 5), ('y', 8))), ('q', (('z', 1), ('x', 1), ('w', 1), ('y', 2))))
                )
        self.assertEqual(g2, (None, None))
        self.assertEqual(post2.to_pairs(),
                (('r', (('z', False), ('x', False), ('w', False), ('y', True))),)
                )

    def test_frame_iter_group_other_d(self) -> None:
        columns = tuple('pqr')
        index = tuple('zxwy')
        records = ((1, 1, False),
                   (1, 1, False),
                   (5, 1, False),
                   (8, 2, True))
        f = Frame.from_records(records, columns=columns, index=index)
        post1, post2 = list(f.iter_group_other(axis=0,
                other=np.array([[1, 1], [0, 0], [1, 1], [0, 0]]),
                fill_value = None
                ))
        self.assertEqual(post1.to_pairs(),
                (('p', (('x', 1), ('y', 8))), ('q', (('x', 1), ('y', 2))), ('r', (('x', False), ('y', True))))
                )
        self.assertEqual(post2.to_pairs(),
                (('p', (('z', 1), ('w', 5))), ('q', (('z', 1), ('w', 1))), ('r', (('z', False), ('w', False))))
                )

    #---------------------------------------------------------------------------
    def test_frame_iter_group_other_items_a(self) -> None:
        columns = tuple('pqr')
        index = tuple('zxwy')
        records = (('A', 1, False),
                   ('A', 2, True),
                   ('B', 1, False),
                   ('B', 2, True))
        f = Frame.from_records(records, columns=columns, index=index)
        # asxis 0 means for other grouping means that they key is a column key
        (g1, post1), (g2, post2) = list(f.iter_group_other_items(
                axis=0, other=(0, 0, 0, 1))
                )

        self.assertEqual(post1.to_pairs(),
                (('p', (('z', 'A'), ('x', 'A'), ('w', 'B'))), ('q', (('z', 1), ('x', 2), ('w', 1))), ('r', (('z', False), ('x', True), ('w', False))))
                )
        self.assertEqual(g1, 0)
        self.assertEqual(post2.to_pairs(),
                (('p', (('y', 'B'),)), ('q', (('y', 2),)), ('r', (('y', True),)))
                )
        self.assertEqual(g2, 1)

        (g3, post3), (g4, post4) = list(f.iter_group_other_items(
                axis=1, other=(1, 0, 1))
                )
        self.assertEqual(post3.to_pairs(),
                (('q', (('z', 1), ('x', 2), ('w', 1), ('y', 2))),)
                )
        self.assertEqual(g3, 0)

        self.assertEqual(post4.to_pairs(),
                (('p', (('z', 'A'), ('x', 'A'), ('w', 'B'), ('y', 'B'))), ('r', (('z', False), ('x', True), ('w', False), ('y', True))))
                )
        self.assertEqual(g4, 1)

    #---------------------------------------------------------------------------
    def test_frame_iter_group_other_array_a(self) -> None:
        columns = tuple('pq')
        index = tuple('zxwy')
        records = ((1, 10),
                   (2, 15),
                   (1, 20),
                   (2, 25))
        f = Frame.from_records(records, columns=columns, index=index)
        # asxis 0 means for other grouping means that they key is a column key
        post1, post2 = list(f.iter_group_other_array(
                axis=0, other=(1, 0, 0, 1))
                )
        self.assertEqual(post1.tolist(),
                [[2, 15], [1, 20]]
                )
        self.assertEqual(post2.tolist(),
                [[1, 10], [2, 25]]
                )

        post3, post4 = list(f.iter_group_other_array(
                axis=1, other=(0, 1))
                )
        self.assertEqual(post3.tolist(),
                [[1], [2], [1], [2]]
                )
        self.assertEqual(post4.tolist(),
                [[10], [15], [20], [25]]
                )

    #---------------------------------------------------------------------------
    def test_frame_iter_group_other_array_items_a(self) -> None:
        columns = tuple('pq')
        index = tuple('zxwy')
        records = ((1, 10),
                   (2, 15),
                   (1, 20),
                   (2, 25))
        f = Frame.from_records(records, columns=columns, index=index)
        # asxis 0 means for other grouping means that they key is a column key
        (g1, post1), (g2, post2) = list(f.iter_group_other_array_items(
                axis=0, other=(1, 0, 0, 1))
                )
        self.assertEqual(g1, 0)
        self.assertEqual(post1.tolist(),
                [[2, 15], [1, 20]]
                )
        self.assertEqual(g2, 1)
        self.assertEqual(post2.tolist(),
                [[1, 10], [2, 25]]
                )

        (g3, post3), (g4, post4) = list(f.iter_group_other_array_items(
                axis=1, other=(0, 1))
                )
        self.assertEqual(g3, 0)
        self.assertEqual(post3.tolist(),
                [[1], [2], [1], [2]]
                )
        self.assertEqual(g4, 1)
        self.assertEqual(post4.tolist(),
                [[10], [15], [20], [25]]
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

    #---------------------------------------------------------------------------

    def test_frame_axis_window_items_a(self) -> None:

        base = np.array([1, 2, 3, 4])
        records = (base * n for n in range(1, 21))

        f1 = Frame.from_records(records,
                columns=list('ABCD'),
                index=self.get_letters(20))

        post0 = tuple(f1._axis_window_items(size=2, axis=0))
        self.assertEqual(len(post0), 19)
        self.assertEqual(post0[0][0], 'b')
        self.assertEqual(post0[0][1].__class__, Frame)
        self.assertEqual(post0[0][1].shape, (2, 4))

        self.assertEqual(post0[-1][0], 't')
        self.assertEqual(post0[-1][1].__class__, Frame)
        self.assertEqual(post0[-1][1].shape, (2, 4))

        post1 = tuple(f1._axis_window_items(size=2, axis=1))
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

        post0 = tuple(f1._axis_window_items(size=2, axis=0, as_array=True))
        self.assertEqual(len(post0), 19)
        self.assertEqual(post0[0][0], 'b')
        self.assertEqual(post0[0][1].__class__, np.ndarray)
        self.assertEqual(post0[0][1].shape, (2, 4))

        self.assertEqual(post0[-1][0], 't')
        self.assertEqual(post0[-1][1].__class__, np.ndarray)
        self.assertEqual(post0[-1][1].shape, (2, 4))

        post1 = tuple(f1._axis_window_items(size=2, axis=1, as_array=True))
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

        post = list(f1.iter_window(size=3))
        self.assertEqual(len(post), 18)
        self.assertTrue(all(f.shape == (3, 4) for f in post))


    def test_frame_iter_window_b(self) -> None:

        base = np.array([1, 2, 3, 4])
        records = (base * n for n in range(1, 21))

        f1 = Frame.from_records(records,
                columns=list('ABCD'),
                index=self.get_letters(20))

        f2 = f1.iter_window(size=8, step=6).reduce.from_label_map({'B': np.sum, 'C': np.min}).to_frame()
        self.assertEqual(f2.to_pairs(),
                (('B', (('h', 72), ('n', 168), ('t', 264))), ('C', (('h', 3), ('n', 21), ('t', 39)))))

        f3 = f1.iter_window(size=8, step=6).reduce.from_label_pair_map({('B', 'B-sum'): np.sum, ('B', 'B-min'): np.min}).to_frame()

        self.assertEqual(f3.to_pairs(),
                (('B-sum', (('h', 72), ('n', 168), ('t', 264))), ('B-min', (('h', 2), ('n', 14), ('t', 26)))))


    #---------------------------------------------------------------------------

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

        for x in f1.iter_tuple(axis=0):
            self.assertTrue(len(x), 4)

        for x in f1.iter_tuple(axis=1):
            self.assertTrue(len(x), 5)


        f2 = f1[['p', 'q']]

        s1 = f2.iter_array(axis=0).apply(np.sum) # type: ignore
        self.assertEqual(list(s1.items()), [('p', 150), ('q', 204)])

        s2 = f2.iter_array(axis=1).apply(np.sum) # type: ignore
        self.assertEqual(list(s2.items()),
                [('w', 3), ('x', 64), ('y', 149), ('z', 138)])

        def sum_if(idx: TLabel, vals: tp.Iterable[int]) -> tp.Optional[int]:
            if idx in ('x', 'z'):
                return np.sum(vals) # type: ignore
            return None

        s3 = f2.iter_array_items(axis=1).apply(sum_if) # type: ignore
        self.assertEqual(list(s3.items()),
                [('w', None), ('x', 64), ('y', None), ('z', 138)])

    #---------------------------------------------------------------------------

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

        self.assertEqual(post[1][0], 'a')
        self.assertEqual(post[1][1].to_pairs(0),
                (('r', (('w', 'a'), ('x', 'b'), ('y', 'c'), ('z', 'd'))),))

        self.assertEqual(post[2][0], False)
        self.assertEqual(post[2][1].to_pairs(0),
                (('s', (('w', False), ('x', True), ('y', False), ('z', True))), ('t', (('w', False), ('x', False), ('y', False), ('z', True)))))

    def test_frame_group_c(self) -> None:
        f = ff.parse('s(10,3)|v(int,str,bool)').assign[0].apply(lambda s: s % 4)
        post1 = tuple(f.iter_group(0, axis=0, drop=True))
        self.assertEqual(len(post1), 2)
        self.assertEqual(post1[0].to_pairs(),
                ((1, ((3, 'zuVU'), (5, 'zJXD'), (6, 'zPAQ'), (7, 'zyps'))), (2, ((3, True), (5, False), (6, True), (7, True))))
                )

        self.assertEqual(post1[1].to_pairs(),
                ((1, ((0, 'zaji'), (1, 'zJnC'), (2, 'zDdR'), (4, 'zKka'), (8, 'zyG5'), (9, 'zvMu'))), (2, ((0, True), (1, False), (2, False), (4, False), (8, True), (9, False))))
                )

        post2 = tuple(f.iter_group_items(0, axis=0, drop=True))
        self.assertEqual(len(post2), 2)
        self.assertEqual(post2[0][0], 0)
        self.assertEqual(post2[0][1].shape, (4, 2))
        self.assertEqual(post2[1][0], 3)
        self.assertEqual(post2[1][1].shape, (6, 2))

    def test_frame_group_d(self) -> None:
        f = ff.parse('s(3,6)|v(bool)')
        post1 = tuple(f.iter_group(0, axis=1, drop=True))
        self.assertEqual(len(post1), 2)
        self.assertEqual(post1[0].shape, (2, 4))
        self.assertEqual(post1[1].shape, (2, 2))

        post2 = tuple(f.iter_group_items(0, axis=1, drop=True))
        self.assertEqual(len(post1), 2)
        self.assertEqual(post2[0][0], False)
        self.assertEqual(post2[0][1].shape, (2, 4))
        self.assertEqual(post2[1][0], True)
        self.assertEqual(post2[1][1].shape, (2, 2))

    #---------------------------------------------------------------------------

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

    #---------------------------------------------------------------------------

    def test_frame_iter_reduce_a(self):
        f1 = ff.parse('s(100,5)|v(int64, int64, int64, int64, int64)')
        f1 = f1.assign[0].apply(lambda s: s % 4)
        f1 = f1.relabel(columns=('a', 'b', 'c', 'd', 'e'))
        f2 = f1.iter_group('a').reduce.from_label_map({'c': np.min, 'b': np.sum, 'd': np.max}).to_frame()
        self.assertEqual(f2.to_pairs(),
                (('c', ((0, -157437), (1, -117006), (2, -171231), (3, -170415))), ('b', ((0, 979722), (1, 260619), (2, -122437), (3, 820941))), ('d', ((0, 195850), (1, 199490), (2, 194249), (3, 197228)))))

    def test_frame_iter_reduce_b(self):
        f1 = ff.parse('s(100,5)|v(int64, int64, int64, int64, int64)')
        f1 = f1.assign[0].apply(lambda s: s % 4)
        f1 = f1.relabel(columns=('a', 'b', 'c', 'd', 'e'))
        f2 = f1.iter_group('a').reduce.from_label_map({'c': np.min, 'b': np.sum, 'd': np.max}).to_frame()
        self.assertEqual(f2.to_pairs(),
                (('c', ((0, -157437), (1, -117006), (2, -171231), (3, -170415))), ('b', ((0, 979722), (1, 260619), (2, -122437), (3, 820941))), ('d', ((0, 195850), (1, 199490), (2, 194249), (3, 197228)))))

    def test_frame_iter_reduce_c(self):
        f1 = ff.parse('s(100,5)|v(int64, int64, int64, int64, int64)')
        f1 = f1.assign[0].apply(lambda s: s % 4)
        f1 = f1.relabel(columns=('a', 'b', 'c', 'd', 'e'))
        f2 = f1.iter_group_array('a').reduce.from_label_map({'c': np.min, 'b': np.sum, 'd': np.max}).to_frame()
        self.assertEqual(f2.to_pairs(),
                (('c', ((0, -157437), (1, -117006), (2, -171231), (3, -170415))), ('b', ((0, 979722), (1, 260619), (2, -122437), (3, 820941))), ('d', ((0, 195850), (1, 199490), (2, 194249), (3, 197228)))))

    def test_frame_iter_reduce_d(self):
        f1 = ff.parse('s(100,5)|v(int64, int64, int64, int64, int64)')
        f1 = f1.assign[0].apply(lambda s: s % 4)
        f1 = f1.relabel(columns=('a', 'b', 'c', 'd', 'e'))

        f2 = f1.iter_group('a').reduce.from_label_pair_map({
                ('a', 'a_sum'): np.sum,
                ('a', 'a_min'): np.min,
                ('d', 'd_sum'): np.sum,
                ('d', 'd_min'): np.min,
                }).to_frame()
        self.assertEqual(f2.to_pairs(),
                (('a_sum', ((0, 0), (1, 21), (2, 40), (3, 81))), ('a_min', ((0, 0), (1, 1), (2, 2), (3, 3))), ('d_sum', ((0, 796595), (1, 608498), (2, 907345), (3, 2018477))), ('d_min', ((0, -171231), (1, -170415), (2, -159324), (3, -112188))))
                )

    def test_frame_iter_reduce_e(self):
        f1 = ff.parse('s(100,5)|v(int64, int64, int64, int64, int64)')
        f1 = f1.assign[0].apply(lambda s: s % 4)
        f1 = f1.relabel(columns=('a', 'b', 'c', 'd', 'e'))
        post = list(f1.iter_group_array('a').reduce.from_label_map({'b': np.sum, 'c': np.min}).values())

        self.assertEqual(post[0].tolist(), [979722, -157437])
        self.assertEqual(post[1].tolist(), [260619, -117006])
        self.assertEqual(post[2].tolist(), [-122437, -171231])
        self.assertEqual(post[3].tolist(), [820941, -170415])


    def test_frame_iter_reduce_f(self):
        f1 = ff.parse('s(100,5)|v(int64, int64, int64, int64, int64)')
        f1 = f1.assign[0].apply(lambda s: s % 4)
        f1 = f1.relabel(columns=('a', 'b', 'c', 'd', 'e'))
        f2 = f1.iter_group('a', drop=True).reduce.from_map_func(lambda s: s[2]).to_frame() # take the third value
        self.assertEqual(f2.to_pairs(),
                (('b', ((0, -171231), (1, -51750), (2, 30628), (3, 5729))), ('c', ((0, 166924), (1, 170440), (2, 84967), (3, 30205))), ('d', ((0, 119909), (1, 172142), (2, 146284), (3, 166924))), ('e', ((0, 172142), (1, 110798), (2, -27771), (3, 170440))))
                )


if __name__ == '__main__':
    import unittest
    unittest.main()
