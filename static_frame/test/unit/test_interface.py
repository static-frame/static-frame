
import unittest

import numpy as np

from doc.source.conf import DOCUMENTED_COMPONENTS
from static_frame.core.container import ContainerBase
from static_frame.core.display_config import DisplayConfig
from static_frame.core.frame import Frame
from static_frame.core.frame import FrameGO
from static_frame.core.interface import InterfaceGroup
from static_frame.core.interface import InterfaceSummary
from static_frame.core.interface import _get_signatures
from static_frame.core.series import Series
from static_frame.core.store import StoreConfig
from static_frame.core.store_filter import StoreFilter
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
            (('Accessor Datetime', 10), ('Accessor Fill Value', 24), ('Accessor Regular Expression', 7), ('Accessor String', 37), ('Accessor Transpose', 24), ('Assignment', 8), ('Attribute', 11), ('Constructor', 34), ('Dictionary-Like', 7), ('Display', 6), ('Exporter', 26), ('Iterator', 224), ('Method', 83), ('Operator Binary', 24), ('Operator Unary', 4), ('Selector', 13))
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


    def test_interface_via_re_signature_no_args(self) -> None:
        inter = InterfaceSummary.to_frame(Series,
                minimized=False,
                max_args=99, # +inf, but keep as int
                )

        self.assertEqual(
            inter.loc[inter['group']==InterfaceGroup.AccessorFillValue, 'signature_no_args'].values.tolist(),
            ['via_fill_value(fill_value).via_T', 'via_fill_value().__add__()', 'via_fill_value().__sub__()', 'via_fill_value().__mul__()', 'via_fill_value().__truediv__()', 'via_fill_value().__floordiv__()', 'via_fill_value().__mod__()', 'via_fill_value().__pow__()', 'via_fill_value().__lshift__()', 'via_fill_value().__rshift__()', 'via_fill_value().__and__()', 'via_fill_value().__xor__()', 'via_fill_value().__or__()', 'via_fill_value().__lt__()', 'via_fill_value().__le__()', 'via_fill_value().__eq__()', 'via_fill_value().__ne__()', 'via_fill_value().__gt__()', 'via_fill_value().__ge__()', 'via_fill_value().__radd__()', 'via_fill_value().__rsub__()', 'via_fill_value().__rmul__()', 'via_fill_value().__rtruediv__()', 'via_fill_value().__rfloordiv__()']
            )

        self.assertEqual(
            inter.loc[inter['group']==InterfaceGroup.AccessorRe, 'signature_no_args'].values.tolist(),
            ['via_re().search()', 'via_re().match()', 'via_re().fullmatch()', 'via_re().split()', 'via_re().findall()', 'via_re().sub()', 'via_re().subn()'])


    def test_interface_get_instance(self) -> None:
        for component in DOCUMENTED_COMPONENTS:
            post = InterfaceSummary.get_instance(component)
            if not isinstance(post, ContainerBase):
                self.assertTrue(isinstance(post, ( # type: ignore
                    DisplayConfig,
                    StoreConfig,
                    StoreFilter,
                    )))

if __name__ == '__main__':
    unittest.main()





