
import unittest
import pydoc


from static_frame.core.interface import InterfaceSummary
from static_frame.core.series import Series
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
        counts = post.iter_group('group').apply(len)
        self.assertEqual(
            counts.to_pairs(),
            (('Attribute', 10), ('Constructor', 26), ('Dictionary-Like', 7), ('Display', 6), ('Exporter', 18), ('Iterator', 80), ('Method', 54), ('Operator Binary', 24), ('Operator Unary', 4), ('Selector', 17))
            )


    def test_interface_summary_c(self) -> None:
        s = Series(['a', 'b', 'c'])
        post = s.interface

        counts = post.iter_group('group').apply(len)
        counts_cls = s.__class__.interface.iter_group('group').apply(len)

        self.assertTrue((counts == counts_cls).all())


    def test_interface_help_a(self) -> None:

        for target in self.get_containers():
            post = pydoc.render_doc(target, renderer=pydoc.plaintext) # type: ignore
            self.assertTrue(len(post) > 0)

if __name__ == '__main__':
    unittest.main()





