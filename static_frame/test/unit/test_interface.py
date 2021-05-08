
import unittest

import numpy as np

from static_frame.core.interface import InterfaceSummary
from static_frame.core.series import Series
from static_frame.core.frame import FrameGO
from static_frame.core.frame import Frame
from static_frame.core.interface import _get_signatures



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
            (('Accessor Datetime', 10), ('Accessor String', 36), ('Accessor Transpose', 23), ('Assignment', 8), ('Attribute', 11), ('Constructor', 30), ('Dictionary-Like', 7), ('Display', 6), ('Exporter', 21), ('Iterator', 224), ('Method', 70), ('Operator Binary', 24), ('Operator Unary', 4), ('Selector', 13))
        )

    def test_interface_summary_c(self) -> None:
        s = Series(['a', 'b', 'c'])
        post = s.interface

        counts = post.iter_group('group').apply(len)
        counts_cls = s.__class__.interface.iter_group('group').apply(len)

        self.assertTrue((counts == counts_cls).all())


    def test_interface_get_signatures_a(self) -> None:

        sig, signa = _get_signatures('__init__', Series.__init__)

        self.assertEqual(sig, '__init__(values, *, index, name, ...)')
        self.assertEqual(signa, '__init__()')

    def test_interface_get_signatures_b(self) -> None:

        sig, signa = _get_signatures('__init__', Series.__init__, max_args=99)

        self.assertEqual(sig, '__init__(values, *, index, name, dtype, index_constructor, own_index)')
        self.assertEqual(signa, '__init__()')

    def test_interface_get_frame_a(self) -> None:

        f1 = InterfaceSummary.to_frame(Series)
        f2 = InterfaceSummary.to_frame(Series, minimized=False, max_args=np.inf)
        self.assertTrue(len(f1) == len(f2))

        self.assertEqual(f1.columns.values.tolist(),
                ['cls_name', 'group', 'doc']
                )
        self.assertEqual(
                f2.columns.values.tolist(),
                ['cls_name', 'group', 'doc', 'reference', 'use_signature', 'is_attr', 'delegate_reference', 'delegate_is_attr', 'signature_no_args']
                )

    def test_interface_assign_a(self) -> None:
        f = Frame.interface.loc[Frame.interface.index.via_str.startswith('assign')]
        # assignmewnt interface is one of the most complex, so we can check the signatures here explicitly
        self.assertEqual(f.index.values.tolist(),
                ['assign[key](value, *, fill_value)', 'assign[key].apply(func, *, fill_value)', 'assign.iloc[key](value, *, fill_value)', 'assign.iloc[key].apply(func, *, fill_value)', 'assign.loc[key](value, *, fill_value)', 'assign.loc[key].apply(func, *, fill_value)', 'assign.bloc[key](value, *, fill_value)', 'assign.bloc[key].apply(func, *, fill_value)'])




if __name__ == '__main__':
    unittest.main()





