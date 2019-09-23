
import unittest
# import numpy as np  # type: ignore


from static_frame.core.interface import InterfaceSummary
# from static_frame.core.series import Series
from static_frame.core.frame import FrameGO
# from static_frame.core.index import Index
# from static_frame.core.index_hierarchy import IndexHierarchy
# from static_frame.core.display import DisplayConfigs


from static_frame.test.test_case import TestCase



class TestUnit(TestCase):

    def test_interface_summary_a(self) -> None:

        for target in self.get_containers():
            t = InterfaceSummary.to_frame(target)
            self.assertTrue(len(t) > 30)


    def test_interface_summary_b(self) -> None:

        post = FrameGO.interface

        counts = post.iter_group('group').apply(lambda x: len(x))

        self.assertEqual(
            counts.to_pairs(),
            (('attribute', 11), ('constructor', 15), ('dict_like', 7), ('display', 4), ('exporter', 8), ('iterator', 60), ('method', 50), ('operator_binary', 24), ('operator_unary', 4), ('selector', 15))
            )



if __name__ == '__main__':
    unittest.main()





