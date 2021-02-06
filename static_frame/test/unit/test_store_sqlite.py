import unittest
from fractions import Fraction
import typing as tp

import numpy as np

from static_frame.core.frame import Frame
from static_frame.test.test_case import TestCase
from static_frame.test.test_case import temp_file
from static_frame.core.index_hierarchy import IndexHierarchy
from static_frame.core.store import StoreConfig
from static_frame.core.store_sqlite import StoreSQLite
from static_frame.core.store import StoreConfigMap


class TestUnit(TestCase):

    def test_store_sqlite_write_a(self) -> None:

        f1 = Frame.from_dict(
                dict(x=(None,-np.inf,np.inf,None), y=(3,4,-5,-3000)),
                index=IndexHierarchy.from_product(('I', 'II'), ('a', 'b')),
                name='f1')
        f2 = Frame.from_dict(
                dict(a=(1,2,3), b=(4,5,6)),
                index=('x', 'y', 'z'),
                name='f2')
        f3 = Frame.from_records(
                ((10.4, 20.1, 50, 60), (50.1, 60.4, -50, -60)),
                index=('p', 'q'),
                columns=IndexHierarchy.from_product(('I', 'II'), ('a', 'b')),
                name='f3')
        f4 = Frame.from_records((
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
                columns=IndexHierarchy.from_product(('I', 'II'), ('a', 'b'), (1, 2)),
                name='f4')

        frames = (f1, f2, f3, f4)

        with temp_file('.sqlite') as fp:

            st1 = StoreSQLite(fp)
            st1.write((f.name, f) for f in frames)

            sheet_names = tuple(st1.labels()) # this will read from file, not in memory
            self.assertEqual(tuple(f.name for f in frames), sheet_names)

            for i, name in enumerate(sheet_names):
                f_src = frames[i]
                config = StoreConfig.from_frame(f_src)
                f_loaded = st1.read(name, config=config)
                self.assertEqualFrames(f_src, f_loaded)

    def test_store_sqlite_write_b(self) -> None:

        f1 = Frame.from_dict(
                dict(
                        x=(Fraction(3,2), Fraction(1,2), Fraction(2,3), Fraction(3,7)),
                        y=(3,4,-5,-3000)),
                index=IndexHierarchy.from_product(('I', 'II'), ('a', 'b')),
                name='f1-dash')

        frames = (f1,)

        with temp_file('.sqlite') as fp:

            st1 = StoreSQLite(fp)
            st1.write((f.name, f) for f in frames)

            config = StoreConfig.from_frame(f1)

            f_loaded = st1.read(f1.name, config=config)

            # for now, Fractions come back as strings
            self.assertEqual(
                    f_loaded['x'].to_pairs(),
                    ((('I', 'a'), '3/2'), (('I', 'b'), '1/2'), (('II', 'a'), '2/3'), (('II', 'b'), '3/7'))
            )

    def test_store_sqlite_write_c(self) -> None:

        f1 = Frame.from_dict(
                dict(
                        x=np.array([1.2, 4.5, 3.2, 6.5], dtype=np.float16),
                        y=(3,4,-5,-3000)),
                index=IndexHierarchy.from_product(('I', 'II'), ('a', 'b')),
                name='f1')

        frames = (f1,)

        with temp_file('.sqlite') as fp:
            st1 = StoreSQLite(fp)
            st1.write((f.name, f) for f in frames)

            config = StoreConfig.from_frame(f1)

            f_loaded = st1.read(f1.name, config=config)

            self.assertAlmostEqualItems(f_loaded['x'].to_pairs(),
                    ((('I', 'a'), 1.2001953125), (('I', 'b'), 4.5), (('II', 'a'), 3.19921875), (('II', 'b'), 6.5))
                    )

    def test_store_sqlite_write_d(self) -> None:

        f1 = Frame.from_dict(
                dict(a=(1,2,3), b=(4,5,6)),
                index=('x', 'y', 'z'),
                name='f2')

        frames = (f1,)

        with temp_file('.sqlite') as fp:

            config = StoreConfig(include_index=False)

            st1 = StoreSQLite(fp)
            st1.write(((f.name, f) for f in frames), config=config)

            f2 = st1.read(f1.name, config=config)

            self.assertEqual(f2.to_pairs(0),
                    (('a', ((0, 1), (1, 2), (2, 3))), ('b', ((0, 4), (1, 5), (2, 6))))
                    )

            # getting the default config
            f3 = st1.read(f1.name, config=None)

            self.assertEqual(f3.to_pairs(0),
                    (('a', ((0, 1), (1, 2), (2, 3))), ('b', ((0, 4), (1, 5), (2, 6))))
                    )


    def test_store_sqlite_write_f(self) -> None:

        f1 = Frame.from_dict(
                dict(a=(1,2,3), b=(4,5,6)),
                index=('x', 'y', 'z'),
                name='f2')

        frames = (f1,)

        with temp_file('.sqlite') as fp:

            config = StoreConfig(include_index=False)

            st1 = StoreSQLite(fp)
            st1.write(((f.name, f) for f in frames), config=config)

            # prove that writing to the same path re-writes
            st2 = StoreSQLite(fp)
            st2.write(((f.name, f) for f in frames), config=config)

            self.assertEqual(list(st2.labels()), ['f2'])

    #---------------------------------------------------------------------------


    def test_store_sqlite_read_many_a(self) -> None:

        f1 = Frame.from_dict(
                dict(x=(1,2,-5,200), y=(3,4,-5,-3000)),
                index=IndexHierarchy.from_product(('I', 'II'), ('a', 'b')),
                name='f1')
        f2 = Frame.from_dict(
                dict(a=(1,2,3), b=(4,5,6)),
                index=('x', 'y', 'z'),
                name='f2')
        f3 = Frame.from_records(
                ((10, 20, 50, 60), (50.0, 60.4, -50, -60)),
                index=('p', 'q'),
                columns=IndexHierarchy.from_product(('I', 'II'), ('a', 'b')),
                name='f3')
        f4 = Frame.from_records((
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
                columns=IndexHierarchy.from_product(('I', 'II'), ('a', 'b'), (1, 2)),
                name='f4')

        frames = (f1, f2, f3, f4)
        config_map_write = StoreConfigMap.from_config(
                StoreConfig(include_index=True, include_columns=True))

        with temp_file('.sqlite') as fp:

            st1 = StoreSQLite(fp)
            st1.write(((f.name, f) for f in frames), config=config_map_write)

            labels = tuple(st1.labels()) # this will read from file, not in memory
            self.assertEqual(tuple(f.name for f in frames), labels)

            config_map_read: tp.Dict[tp.Hashable, StoreConfig] = {}
            for i, name in enumerate(labels):
                f_src = frames[i]
                c = StoreConfig(
                        index_depth=f_src.index.depth,
                        columns_depth=f_src.columns.depth
                        )
                config_map_read[name] = c

            for i, f_loaded in enumerate(st1.read_many(labels, config=config_map_read)):
                f_src = frames[i]
                self.assertEqualFrames(f_src, f_loaded, compare_dtype=False)


if __name__ == '__main__':
    unittest.main()





