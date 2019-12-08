
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

        with temp_file('.xlsx') as fp:

            st1 = StoreXLSX(fp)
            st1.write((f.name, f) for f in frames)

            # import ipdb; ipdb.set_trace()
            sheet_names = tuple(st1.labels()) # this will read from file, not in memory
            self.assertEqual(tuple(f.name for f in frames), sheet_names)

            for i, name in enumerate(sheet_names):
                f_src = frames[i]
                f_loaded = st1.read(name,
                        index_depth=f_src.index.depth,
                        columns_depth=f_src.columns.depth
                        )
                self.assertEqualFrames(f_src, f_loaded)




    def test_store_xlsx_write_b(self) -> None:

        f1 = Frame.from_records(
                ((None, np.nan, 50, 'a'), (None, -np.inf, -50, 'b'), (None, 60.4, -50, 'c')),
                index=('p', 'q', 'r'),
                columns=IndexHierarchy.from_product(('I', 'II'), ('a', 'b')),
                )

        with temp_file('.xlsx') as fp:

            st = StoreXLSX(fp)
            st.write(((None, f1),))

            f2 = st.read(index_depth=f1.index.depth, columns_depth=f1.columns.depth)

            # just a sample column for now
            self.assertEqual(
                    f1[HLoc[('II', 'a')]].values.tolist(),
                    f2[HLoc[('II', 'a')]].values.tolist() )

            self.assertEqualFrames(f1, f2)


    def test_store_xlsx_read_a(self) -> None:
        f1 = Frame([1, 2, 3], index=('a', 'b', 'c'), columns=('x',))

        with temp_file('.xlsx') as fp:

            st = StoreXLSX(fp)
            st.write(((None, f1),), include_index=False)

            f2 = st.read(index_depth=0, columns_depth=f1.columns.depth)

        self.assertTrue((f1.values == f2.values).all())
        self.assertEqual(f2.to_pairs(0),
                (('x', ((0, 1), (1, 2), (2, 3))),)
                )

        # TODO: same test with hierarchical columns, index


if __name__ == '__main__':
    unittest.main()



