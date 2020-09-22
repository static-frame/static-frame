
import unittest

import numpy as np


from static_frame.core.frame import Frame
from static_frame.core.index_hierarchy import IndexHierarchy
from static_frame.core.hloc import HLoc
# from static_frame.core.series import Series

from static_frame.test.test_case import TestCase
from static_frame.test.test_case import temp_file

# from static_frame.test.test_case import skip_win
# from static_frame.core.exception import ErrorInitStore

from static_frame.core.store_xlsx import StoreXLSX
from static_frame.core.store import StoreConfig
from static_frame.core.store import StoreConfigMap
from static_frame.core.store_filter import StoreFilter


class TestUnit(TestCase):


    def test_store_xlsx_write_a(self) -> None:

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
        config_map = StoreConfigMap.from_config(
                StoreConfig(include_index=True, include_columns=True))

        with temp_file('.xlsx') as fp:

            st1 = StoreXLSX(fp)
            st1.write(((f.name, f) for f in frames), config=config_map)

            # import ipdb; ipdb.set_trace()
            sheet_names = tuple(st1.labels()) # this will read from file, not in memory
            self.assertEqual(tuple(f.name for f in frames), sheet_names)

            for i, name in enumerate(sheet_names):
                f_src = frames[i]
                c = StoreConfig(
                        index_depth=f_src.index.depth,
                        columns_depth=f_src.columns.depth
                        )
                f_loaded = st1.read(name, config=c)
                self.assertEqualFrames(f_src, f_loaded, compare_dtype=False)




    def test_store_xlsx_write_b(self) -> None:

        f1 = Frame.from_records(
                ((None, np.nan, 50, 'a'), (None, -np.inf, -50, 'b'), (None, 60.4, -50, 'c')),
                index=('p', 'q', 'r'),
                columns=IndexHierarchy.from_product(('I', 'II'), ('a', 'b')),
                )

        config_map = StoreConfigMap.from_config(
                StoreConfig(include_index=True, include_columns=True))

        with temp_file('.xlsx') as fp:

            st = StoreXLSX(fp)
            st.write(((None, f1),), config=config_map)

            c = StoreConfig(
                    index_depth=f1.index.depth,
                    columns_depth=f1.columns.depth
                    )
            f2 = st.read(None, config=c)

            # just a sample column for now
            self.assertEqual(
                    f1[HLoc[('II', 'a')]].values.tolist(),
                    f2[HLoc[('II', 'a')]].values.tolist() )

            self.assertEqualFrames(f1, f2)


    def test_store_xlsx_read_a(self) -> None:
        f1 = Frame.from_elements([1, 2, 3], index=('a', 'b', 'c'), columns=('x',))

        config_map = StoreConfigMap.from_config(
                StoreConfig(include_index=False, include_columns=True))

        with temp_file('.xlsx') as fp:

            st = StoreXLSX(fp)
            st.write(((None, f1),), config=config_map)

            c = StoreConfig(
                    index_depth=0,
                    columns_depth=f1.columns.depth
                    )
            f2 = st.read(None, config=c)

        self.assertTrue((f1.values == f2.values).all())
        self.assertEqual(f2.to_pairs(0),
                (('x', ((0, 1), (1, 2), (2, 3))),)
                )


    def test_store_xlsx_read_b(self) -> None:
        index = IndexHierarchy.from_product(('left', 'right'), ('up', 'down'))
        columns = IndexHierarchy.from_labels(((100, -5, 20),))

        f1 = Frame.from_elements([1, 2, 3, 4], index=index, columns=columns)

        config_map = StoreConfigMap.from_config(
                StoreConfig(include_index=False, include_columns=True))

        with temp_file('.xlsx') as fp:

            st = StoreXLSX(fp)
            st.write(((None, f1),), config=config_map)

            c = StoreConfig(
                    index_depth=0,
                    columns_depth=f1.columns.depth
                    )
            f2 = st.read(None, config=c)

        self.assertTrue((f1.values == f2.values).all())
        self.assertEqual(f2.to_pairs(0),
                (((100, -5, 20), ((0, 1), (1, 2), (2, 3), (3, 4))),)
                )


    def test_store_xlsx_read_c(self) -> None:
        index = IndexHierarchy.from_product(('left', 'right'), ('up', 'down'))
        columns = IndexHierarchy.from_labels(((100, -5, 20),))

        f1 = Frame.from_elements([1, 2, 3, 4], index=index, columns=columns)

        config_map = StoreConfigMap.from_config(
                StoreConfig(include_index=True, include_columns=False))

        with temp_file('.xlsx') as fp:

            st = StoreXLSX(fp)
            st.write(((None, f1),), config=config_map[None])

            c = StoreConfig(
                    index_depth=f1.index.depth,
                    columns_depth=0
                    )
            f2 = st.read(None, config=c)

        self.assertTrue((f1.values == f2.values).all())
        self.assertEqual(f2.to_pairs(0),
                ((0, ((('left', 'up'), 1), (('left', 'down'), 2), (('right', 'up'), 3), (('right', 'down'), 4))),)
                )


    def test_store_xlsx_read_d(self) -> None:

        f1 = Frame.from_records(
                ((10, 20, 50, 60), (50.0, 60.4, -50, -60)),
                index=('p', 'q'),
                columns=('a', 'b', 'c', 'd'),
                name='f1')

        sc1 = StoreConfig(include_index=False, include_columns=True)
        sc2 = StoreConfig(columns_depth=0, index_depth=0)

        with temp_file('.xlsx') as fp:

            st = StoreXLSX(fp)
            st.write(((None, f1),), config=sc1)


            f2 = st.read(None) #  get default config
            self.assertEqual(f2.to_pairs(0),
                    (('a', ((0, 10), (1, 50))), ('b', ((0, 20.0), (1, 60.4))), ('c', ((0, 50), (1, -50))), ('d', ((0, 60), (1, -60)))))

            f3 = st.read(None, config=sc2)
            self.assertEqual(f3.to_pairs(0),
                    ((0, ((0, 'a'), (1, 10), (2, 50))), (1, ((0, 'b'), (1, 20), (2, 60.4))), (2, ((0, 'c'), (1, 50), (2, -50))), (3, ((0, 'd'), (1, 60), (2, -60)))))


    def test_store_xlsx_read_e(self) -> None:

        f1 = Frame.from_records(
                ((np.inf, np.inf), (-np.inf, -np.inf)),
                index=('p', 'q'),
                columns=('a', 'b'),
                name='f1')

        sc1 = StoreConfig(columns_depth=1, index_depth=1)

        with temp_file('.xlsx') as fp:

            st = StoreXLSX(fp)
            st.write(((None, f1),))

            f1 = st.read(None, config=sc1, store_filter=None)
            self.assertEqual(f1.to_pairs(0),
                    (('a', (('p', 'inf'), ('q', '-inf'))), ('b', (('p', 'inf'), ('q', '-inf')))))

            f2 = st.read(None, config=sc1, store_filter=StoreFilter())
            self.assertEqual(f2.to_pairs(0),
                    (('a', (('p', np.inf), ('q', -np.inf))),
                    ('b', (('p', np.inf), ('q', -np.inf)))))



    #---------------------------------------------------------------------------

    def test_dtype_to_writer_attr(self) -> None:
        attr1, switch1 = StoreXLSX._dtype_to_writer_attr(
                np.array(('2020', '2021'), dtype=np.datetime64).dtype)

        self.assertEqual(attr1, 'write')
        self.assertEqual(switch1, True)

        attr2, switch2 = StoreXLSX._dtype_to_writer_attr(
                np.array(('2020-01-01', '2021-01-01'), dtype=np.datetime64).dtype)

        self.assertEqual(attr2, 'write')
        self.assertEqual(switch2, True)

if __name__ == '__main__':
    unittest.main()



