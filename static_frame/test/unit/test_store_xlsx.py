
import unittest

from static_frame.core.frame import Frame
from static_frame.core.index_hierarchy import IndexHierarchy
# from static_frame.core.series import Series

from static_frame.test.test_case import TestCase
from static_frame.test.test_case import temp_file

# from static_frame.test.test_case import skip_win
# from static_frame.core.exception import ErrorInitStore

from static_frame.core.store_xlsx import StoreXLSX


class TestUnit(TestCase):


    def test_store_xlsx_write_a(self) -> None:

        f1 = Frame.from_dict(
                dict(a=(1,2,-5,200), b=(3,4,-5,-3000)),
                index=IndexHierarchy.from_product(('I', 'II'), ('a', 'b')),
                name='foo')
        f2 = Frame.from_dict(
                dict(a=(1,2,3), b=(4,5,6)),
                index=('x', 'y', 'z'),
                name='bar')
        f3 = Frame.from_dict(
                dict(a=(10,20), b=(50,60)),
                index=('p', 'q'),
                name='baz')

        with temp_file('.xlsx') as fp:

            st = StoreXLSX(fp)
            st.write((f.name, f) for f in (f1, f2, f3))


            # import ipdb; ipdb.set_trace()


            pass


if __name__ == '__main__':
    unittest.main()



# x x I I II II
# x x a b a  b
# 1 A 0 0 0  0
# 1 B 0 0 0  0