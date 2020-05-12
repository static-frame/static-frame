
import unittest
from static_frame.test.test_case import TestCase
from static_frame.core.frame import Frame
from static_frame.test.test_case import temp_file
from static_frame.core.index_auto import IndexAutoFactory

class TestUnit(TestCase):

    def test_exceed_columns(self) -> None:

        f1 = Frame.from_element('x', index='x', columns=range(16384))

        with temp_file('.xlsx') as fp:

            with self.assertRaises(RuntimeError):
                # with the index, the limit is exceeded
                f1.to_xlsx(fp, include_index=True)


            f1.to_xlsx(fp, include_index=False)
            f2 = Frame.from_xlsx(fp, index_depth=0, columns_depth=1)
            # need to remove index on original for appropriate comparison
            self.assertEqualFrames(f1.relabel(index=IndexAutoFactory), f2)


    def test_exceed_rows(self) -> None:

        f1 = Frame.from_element('x', index=range(1048576), columns='x')

        with temp_file('.xlsx') as fp:

            with self.assertRaises(RuntimeError):
                # with the index, the limit is exceeded
                f1.to_xlsx(fp, include_columns=True)

            # NOTE: it takes almost 60s to write this file, so we will skip testing it
            # f1.to_xlsx(fp, include_columns=False)
            # f2 = Frame.from_xlsx(fp, index_depth=1, columns_depth=0)
            # # need to remove index on original for appropriate comparison
            # self.assertEqualFrames(f1.relabel(columns=IndexAutoFactory), f2)



if __name__ == '__main__':
    unittest.main()

