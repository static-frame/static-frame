
import unittest
# import numpy as np  # type: ignore


from static_frame.core.interface import InterfaceSummary
# from static_frame.core.series import Series
from static_frame.core.frame import FrameGO
# from static_frame.core.index import Index
# from static_frame.core.index_hierarchy import IndexHierarchy
from static_frame.core.display import DisplayConfigs


from static_frame.test.test_case import TestCase



class TestUnit(TestCase):

    def test_interface_summary_a(self) -> None:

        for target in self.get_containers():
            t = InterfaceSummary.to_frame(target)
            self.assertTrue(len(t) > 30)


    def test_interface_summarY_b(self) -> None:

        post = FrameGO.interface
        print(post.display(DisplayConfigs.UNBOUND))



if __name__ == '__main__':
    unittest.main()





