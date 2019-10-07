
import unittest

# import numpy as np  # type: ignore


from static_frame.core.frame import Frame
from static_frame.core.index_hierarchy import IndexHierarchy
# from static_frame.core.hloc import HLoc
# from static_frame.core.series import Series

from static_frame.test.test_case import TestCase
from static_frame.test.test_case import temp_file


from static_frame.core.store_hdf5 import StoreHDF5


class TestUnit(TestCase):


    def test_store_hdf5_write_a(self) -> None:

        f1 = Frame.from_dict(
                dict(x=(1,2,-5,200), y=(3,4,-5,-3000)),
                index=IndexHierarchy.from_product(('I', 'II'), ('a', 'b')),
                name='f1')
        f2 = Frame.from_dict(
                dict(a=(1,2,3), b=(4,5,6)),
                index=('x', 'y', 'z'),
                name='f2')
        # f3 = Frame.from_records(
        #         ((10, 20, 50, 60), (50.0, 60.4, -50, -60)),
        #         index=('p', 'q'),
        #         columns=IndexHierarchy.from_product(('I', 'II'), ('a', 'b')),
        #         name='f3')
        # f4 = Frame.from_records((
        #         (10, 20, 50, False, 10, 20, 50, False),
        #         (50.0, 60.4, -50, True, 50.0, 60.4, -50, True),
        #         (234, 44452, 0, False, 234, 44452, 0, False),
        #         (4, -4, 2000, True, 4, -4, 2000, True),
        #         (10, 20, 50, False, 10, 20, 50, False),
        #         (50.0, 60.4, -50, True, 50.0, 60.4, -50, True),
        #         (234, 44452, 0, False, 234, 44452, 0, False),
        #         (4, -4, 2000, True, 4, -4, 2000, True),
        #         ),
        #         index=IndexHierarchy.from_product(('top', 'bottom'), ('far', 'near'), ('left', 'right')),
        #         columns=IndexHierarchy.from_product(('I', 'II'), ('a', 'b'), (1, 2)),
        #         name='f4')

        frames = (f1, f2,)

        with temp_file('.hdf5') as fp:

            st1 = StoreHDF5(fp)
            st1.write((f.name, f) for f in frames)

            sheet_names = tuple(st1.labels()) # this will read from file, not in memory
            self.assertEqual(tuple(f.name for f in frames), sheet_names)

            # import ipdb; ipdb.set_trace()
            # pass

            for i, name in enumerate(sheet_names):
                f_src = frames[i]
                f_loaded = st1.read(name,
                        index_depth=f_src.index.depth,
                        columns_depth=f_src.columns.depth
                        )
            #     self.assertEqualFrames(f_src, f_loaded)
                print(name)
                print(f_loaded)


if __name__ == '__main__':
    unittest.main()



