from __future__ import annotations

import frame_fixtures as ff
import numpy as np
import typing_extensions as tp

from static_frame.core.frame import Frame
from static_frame.core.hloc import HLoc
from static_frame.core.index_hierarchy import IndexHierarchy
from static_frame.core.store_config import StoreConfig, StoreConfigMap
from static_frame.core.store_filter import StoreFilter
from static_frame.core.store_xlsx import StoreXLSX
from static_frame.core.util import STORE_LABEL_DEFAULT, TLabel
from static_frame.test.test_case import TestCase, temp_file


class TestUnit(TestCase):
    def test_store_xlsx_write_a(self) -> None:
        f1 = Frame.from_dict(
            dict(x=(1, 2, -5, 200), y=(3, 4, -5, -3000)),
            index=IndexHierarchy.from_product(('I', 'II'), ('a', 'b')),
            name='f1',
        )
        f2 = Frame.from_dict(
            dict(a=(1, 2, 3), b=(4, 5, 6)), index=('x', 'y', 'z'), name='f2'
        )
        f3 = Frame.from_records(
            ((10, 20, 50, 60), (50.0, 60.4, -50, -60)),
            index=('p', 'q'),
            columns=IndexHierarchy.from_product(('I', 'II'), ('a', 'b')),
            name='f3',
        )
        f4 = Frame.from_records(
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
            name='f4',
        )

        frames = (f1, f2, f3, f4)
        config_map = StoreConfigMap.from_config(
            StoreConfig(include_index=True, include_columns=True)
        )

        with temp_file('.xlsx') as fp:
            st1 = StoreXLSX(fp)
            st1.write(((f.name, f) for f in frames), config=config_map)

            sheet_names = tuple(st1.labels())  # this will read from file, not in memory
            self.assertEqual(tuple(f.name for f in frames), sheet_names)

            for i, name in enumerate(sheet_names):
                f_src = frames[i]
                c = StoreConfig(
                    index_depth=f_src.index.depth, columns_depth=f_src.columns.depth
                )
                f_loaded = st1.read(name, config=c)
                self.assertEqualFrames(f_src, f_loaded, compare_dtype=False)

    def test_store_xlsx_write_b(self) -> None:
        f1 = Frame.from_records(
            (
                (None, np.nan, 50, 'a'),
                (None, -np.inf, -50, 'b'),
                (None, 60.4, -50, 'c'),
            ),
            index=('p', 'q', 'r'),
            columns=IndexHierarchy.from_product(('I', 'II'), ('a', 'b')),
        )

        config_map = StoreConfigMap.from_config(
            StoreConfig(include_index=True, include_columns=True)
        )

        with temp_file('.xlsx') as fp:
            st = StoreXLSX(fp)
            st.write(((STORE_LABEL_DEFAULT, f1),), config=config_map)

            c = StoreConfig(index_depth=f1.index.depth, columns_depth=f1.columns.depth)
            f2 = st.read(STORE_LABEL_DEFAULT, config=c)

            # just a sample column for now
            self.assertEqual(
                f1[HLoc[('II', 'a')]].values.tolist(),
                f2[HLoc[('II', 'a')]].values.tolist(),
            )

            self.assertEqualFrames(f1, f2)

    def test_store_xlsx_read_a(self) -> None:
        f1 = Frame.from_elements([1, 2, 3], index=('a', 'b', 'c'), columns=('x',))

        config_map = StoreConfigMap.from_config(
            StoreConfig(include_index=False, include_columns=True)
        )

        with temp_file('.xlsx') as fp:
            st = StoreXLSX(fp)
            st.write(((STORE_LABEL_DEFAULT, f1),), config=config_map)

            c = StoreConfig(index_depth=0, columns_depth=f1.columns.depth)
            f2 = st.read(STORE_LABEL_DEFAULT, config=c)

        self.assertTrue((f1.values == f2.values).all())
        self.assertEqual(f2.to_pairs(), (('x', ((0, 1), (1, 2), (2, 3))),))

    def test_store_xlsx_read_b(self) -> None:
        index = IndexHierarchy.from_product(('left', 'right'), ('up', 'down'))
        columns = IndexHierarchy.from_labels(((100, -5, 20),))

        f1 = Frame.from_elements([1, 2, 3, 4], index=index, columns=columns)

        config_map = StoreConfigMap.from_config(
            StoreConfig(include_index=False, include_columns=True)
        )

        with temp_file('.xlsx') as fp:
            st = StoreXLSX(fp)
            st.write(((STORE_LABEL_DEFAULT, f1),), config=config_map)

            c = StoreConfig(index_depth=0, columns_depth=f1.columns.depth)
            f2 = st.read(STORE_LABEL_DEFAULT, config=c)

        self.assertTrue((f1.values == f2.values).all())
        self.assertEqual(
            f2.to_pairs(), (((100, -5, 20), ((0, 1), (1, 2), (2, 3), (3, 4))),)
        )

    def test_store_xlsx_read_c(self) -> None:
        index = IndexHierarchy.from_product(('left', 'right'), ('up', 'down'))
        columns = IndexHierarchy.from_labels(((100, -5, 20),))

        f1 = Frame.from_elements([1, 2, 3, 4], index=index, columns=columns)

        config_map = StoreConfigMap.from_config(
            StoreConfig(include_index=True, include_columns=False)
        )

        with temp_file('.xlsx') as fp:
            st = StoreXLSX(fp)
            st.write(((STORE_LABEL_DEFAULT, f1),), config=config_map[None])

            c = StoreConfig(index_depth=f1.index.depth, columns_depth=0)
            f2 = st.read(STORE_LABEL_DEFAULT, config=c)

        self.assertTrue((f1.values == f2.values).all())
        self.assertEqual(
            f2.to_pairs(),
            (
                (
                    0,
                    (
                        (('left', 'up'), 1),
                        (('left', 'down'), 2),
                        (('right', 'up'), 3),
                        (('right', 'down'), 4),
                    ),
                ),
            ),
        )

    def test_store_xlsx_read_d(self) -> None:
        f1 = Frame.from_records(
            ((10, 20, 50, 60), (50.0, 60.4, -50, -60)),
            index=('p', 'q'),
            columns=('a', 'b', 'c', 'd'),
            name='f1',
        )

        sc1 = StoreConfig(include_index=False, include_columns=True)
        sc2 = StoreConfig(columns_depth=0, index_depth=0)

        with temp_file('.xlsx') as fp:
            st = StoreXLSX(fp)
            st.write(((STORE_LABEL_DEFAULT, f1),), config=sc1)

            f2 = st.read(STORE_LABEL_DEFAULT)  #  get default config
            self.assertEqual(
                f2.to_pairs(),
                (
                    ('a', ((0, 10), (1, 50))),
                    ('b', ((0, 20.0), (1, 60.4))),
                    ('c', ((0, 50), (1, -50))),
                    ('d', ((0, 60), (1, -60))),
                ),
            )

            f3 = st.read(STORE_LABEL_DEFAULT, config=sc2)
            self.assertEqual(
                f3.to_pairs(),
                (
                    (0, ((0, 'a'), (1, 10), (2, 50))),
                    (1, ((0, 'b'), (1, 20), (2, 60.4))),
                    (2, ((0, 'c'), (1, 50), (2, -50))),
                    (3, ((0, 'd'), (1, 60), (2, -60))),
                ),
            )

    def test_store_xlsx_read_e(self) -> None:
        f1 = Frame.from_records(
            ((np.inf, np.inf), (-np.inf, -np.inf)),
            index=('p', 'q'),
            columns=('a', 'b'),
            name='f1',
        )

        sc1 = StoreConfig(columns_depth=1, index_depth=1)

        with temp_file('.xlsx') as fp:
            st = StoreXLSX(fp)
            st.write(((STORE_LABEL_DEFAULT, f1),))

            f1 = st.read(STORE_LABEL_DEFAULT, config=sc1, store_filter=None)
            self.assertEqual(
                f1.to_pairs(),
                (
                    ('a', (('p', 'inf'), ('q', '-inf'))),
                    ('b', (('p', 'inf'), ('q', '-inf'))),
                ),
            )

            f2 = st.read(STORE_LABEL_DEFAULT, config=sc1, store_filter=StoreFilter())
            self.assertEqual(
                f2.to_pairs(),
                (
                    ('a', (('p', np.inf), ('q', -np.inf))),
                    ('b', (('p', np.inf), ('q', -np.inf))),
                ),
            )

    def test_store_xlsx_read_many_a(self) -> None:
        f1 = Frame.from_dict(
            dict(x=(1, 2, -5, 200), y=(3, 4, -5, -3000)),
            index=IndexHierarchy.from_product(('I', 'II'), ('a', 'b')),
            name='f1',
        )
        f2 = Frame.from_dict(
            dict(a=(1, 2, 3), b=(4, 5, 6)), index=('x', 'y', 'z'), name='f2'
        )
        f3 = Frame.from_records(
            ((10, 20, 50, 60), (50.0, 60.4, -50, -60)),
            index=('p', 'q'),
            columns=IndexHierarchy.from_product(('I', 'II'), ('a', 'b')),
            name='f3',
        )
        f4 = Frame.from_records(
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
            name='f4',
        )

        frames = (f1, f2, f3, f4)
        config_map_write = StoreConfigMap.from_config(
            StoreConfig(include_index=True, include_columns=True)
        )

        with temp_file('.xlsx') as fp:
            st1 = StoreXLSX(fp)
            st1.write(((f.name, f) for f in frames), config=config_map_write)

            sheet_names = tuple(st1.labels())  # this will read from file, not in memory
            self.assertEqual(tuple(f.name for f in frames), sheet_names)

            config_map_read: tp.Dict[TLabel, StoreConfig] = {}
            for i, name in enumerate(sheet_names):
                f_src = frames[i]
                c = StoreConfig(
                    index_depth=f_src.index.depth, columns_depth=f_src.columns.depth
                )
                config_map_read[name] = c

            for i, f_loaded in enumerate(
                st1.read_many(sheet_names, config=config_map_read)
            ):
                f_src = frames[i]
                self.assertEqualFrames(f_src, f_loaded, compare_dtype=False)

    def test_store_xlsx_read_many_b(self) -> None:
        records = (
            (2, 2, 'a', False, None),
            (30, 73, 'd', True, None),
            (None, None, None, None, None),
            (None, None, None, None, None),
        )
        f1 = Frame.from_records(records, columns=('p', 'q', 'r', 's', 't'))

        with temp_file('.xlsx') as fp:
            f1.to_xlsx(fp, label='f1', include_index=False, include_columns=False)

            st1 = StoreXLSX(fp)
            c = StoreConfig(
                index_depth=1,  # force coverage
                columns_depth=0,
                trim_nadir=True,
            )
            f2 = next(st1.read_many(('f1',), config=c))
            self.assertEqual(f2.shape, (2, 3))

    def test_store_xlsx_read_many_c(self) -> None:
        records = (
            (2, 2, 'a', False, None),
            (30, 73, 'd', True, None),
            (None, None, None, None, None),
            (None, None, None, None, None),
        )
        f1 = Frame.from_records(records, columns=('p', 'q', 'r', 's', None))

        with temp_file('.xlsx') as fp:
            f1.to_xlsx(fp, label='f1', include_index=False, include_columns=True)

            st1 = StoreXLSX(fp)
            c = StoreConfig(
                index_depth=0,
                columns_depth=1,
                trim_nadir=True,
            )
            f2 = next(st1.read_many(('f1',), config=c))
            self.assertEqual(f2.shape, (2, 4))

    def test_store_xlsx_read_many_d(self) -> None:
        records = (
            (2, 2, 'a', False, None),
            (30, 73, 'd', True, None),
            (None, None, None, None, None),
            (None, None, None, None, None),
        )
        columns = IndexHierarchy.from_labels(
            (('a', 1), ('a', 2), ('b', 1), ('b', 2), (None, None))
        )
        f1 = Frame.from_records(records, columns=columns)

        with temp_file('.xlsx') as fp:
            f1.to_xlsx(fp, label='f1', include_index=False, include_columns=True)

            st1 = StoreXLSX(fp)
            c = StoreConfig(
                index_depth=0,
                columns_depth=2,
                trim_nadir=True,
            )
            f2 = next(st1.read_many(('f1',), config=c))
            self.assertEqual(f2.shape, (2, 4))
            self.assertEqual(
                f2.to_pairs(),
                (
                    (('a', 1), ((0, 2), (1, 30))),
                    (('a', 2), ((0, 2), (1, 73))),
                    (('b', 1), ((0, 'a'), (1, 'd'))),
                    (('b', 2), ((0, False), (1, True))),
                ),
            )

    def test_store_xlsx_read_many_e(self) -> None:
        records = (
            (2, 2, 'a', False, None),
            (30, 73, 'd', True, None),
            (None, None, None, None, None),
            (None, None, None, None, None),
        )
        f1 = Frame.from_records(records, columns=('p', 'q', 'r', 's', None))

        with temp_file('.xlsx') as fp:
            f1.to_xlsx(fp, label='f1', include_index=False, include_columns=True)

            st1 = StoreXLSX(fp)
            c = StoreConfig(
                index_depth=0,
                columns_depth=1,
                trim_nadir=True,
            )
            # NOTE: if store_filter is None, None is not properly identified as a nadir-area entity and trim_nadir does not drop any rows or columns here
            f2 = next(st1.read_many(('f1',), store_filter=None, config=c))
            self.assertEqual(f2.shape, (4, 5))

    def test_store_xlsx_read_many_f(self) -> None:
        records = (
            (2, 2, 'a', False, None),
            (30, 73, 'd', True, None),
            (None, None, None, None, None),
            (None, None, None, None, None),
        )
        f1 = Frame.from_records(records, columns=('p', 'q', 'r', 's', 't'))

        with temp_file('.xlsx') as fp:
            f1.to_xlsx(fp, label='f1', include_index=False, include_columns=False)

            st1 = StoreXLSX(fp)
            c = StoreConfig(
                index_depth=3,  # force coverage
                columns_depth=0,
                trim_nadir=True,
            )
            f2 = next(st1.read_many(('f1',), config=c))
            self.assertEqual(f2.shape, (2, 1))
            self.assertEqual(
                f2.to_pairs(), ((0, (((2, 2, 'a'), False), ((30, 73, 'd'), True))),)
            )

    # ---------------------------------------------------------------------------

    def test_dtype_to_writer_attr(self) -> None:
        attr1, switch1 = StoreXLSX._dtype_to_writer_attr(
            np.array(('2020', '2021'), dtype=np.datetime64).dtype
        )

        self.assertEqual(attr1, 'write')
        self.assertEqual(switch1, True)

        attr2, switch2 = StoreXLSX._dtype_to_writer_attr(
            np.array(('2020-01-01', '2021-01-01'), dtype=np.datetime64).dtype
        )

        self.assertEqual(attr2, 'write')
        self.assertEqual(switch2, True)

    # ---------------------------------------------------------------------------

    def test_store_xlsx_frame_filter_a(self) -> None:
        f1 = ff.parse('s(4,6)|v(int,int,bool)|i(I,str)|c(I,str)').rename('a')
        f2 = ff.parse('s(4,6)|v(bool,str,float)|i(I,str)|c(I,str)').rename('b')
        f3 = ff.parse('s(4,6)|v(str)|i(I,str)|c(I,str)').rename('c')

        def read_frame_filter(l, f):
            if l in ('a', 'c'):
                return f.iloc[:2, :3]
            return f

        config = StoreConfig(read_frame_filter=read_frame_filter)

        with temp_file('.xlsx') as fp:
            st1 = StoreXLSX(fp)
            st1.write(((f.name, f) for f in (f1, f2, f3)))

            st2 = StoreXLSX(fp)
            post1 = [st2.read(l, config=config).shape for l in ('a', 'b', 'c')]
            self.assertEqual(post1, [(2, 3), (4, 7), (2, 3)])


if __name__ == '__main__':
    import unittest

    unittest.main()
