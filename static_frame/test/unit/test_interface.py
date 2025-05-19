from __future__ import annotations

import numpy as np

from static_frame import FillValueAuto
from static_frame.core.batch import Batch
from static_frame.core.bus import Bus
from static_frame.core.frame import Frame
from static_frame.core.frame import FrameGO
from static_frame.core.index import Index
from static_frame.core.index_datetime import IndexDate
from static_frame.core.index_datetime import IndexDateGO
from static_frame.core.index_hierarchy import IndexHierarchy
from static_frame.core.interface import DOCUMENTED_COMPONENTS
from static_frame.core.interface import InterfaceGroup
from static_frame.core.interface import InterfaceSummary
from static_frame.core.interface import _get_signatures
from static_frame.core.interface import valid_argument_types
from static_frame.core.quilt import Quilt
from static_frame.core.series import Series
from static_frame.core.type_clinic import Require
from static_frame.core.www import WWW
from static_frame.core.yarn import Yarn
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
            tuple((str(x), int(y)) for x, y in counts.to_pairs()),
            (('Accessor Datetime', 23), ('Accessor Fill Value', 26), ('Accessor Hashlib', 10), ('Accessor Reduce', 20), ('Accessor Regular Expression', 7), ('Accessor String', 39), ('Accessor Transpose', 24), ('Accessor Type Clinic', 5), ('Accessor Values', 3), ('Assignment', 16), ('Attribute', 12), ('Constructor', 37), ('Dictionary-Like', 7), ('Display', 6), ('Exporter', 31), ('Iterator', 396), ('Method', 106), ('Operator Binary', 24), ('Operator Unary', 5), ('Selector', 13))
            )

    def test_interface_summary_c(self) -> None:
        s = Series(['a', 'b', 'c'])
        post = s.interface

        counts = post.iter_group('group').apply(len)
        counts_cls = s.__class__.interface.iter_group('group').apply(len)

        self.assertTrue((counts == counts_cls).all())

    #---------------------------------------------------------------------------
    def test_interface_get_signatures_a(self) -> None:

        sig, signa = _get_signatures('__init__', Series.__init__)
        self.assertEqual(sig, '__init__(values, /, *, index, name, ...)')
        self.assertEqual(signa, '__init__()')

    def test_interface_get_signatures_b(self) -> None:

        sig, signa = _get_signatures('__init__', Series.__init__, max_args=99)

        self.assertEqual(sig, '__init__(values, /, *, index, name, dtype, index_constructor, own_index)')
        self.assertEqual(signa, '__init__()')

    def test_interface_get_signatures_c(self) -> None:

        sig, signa = _get_signatures('sum', sum, max_args=99)
        self.assertEqual(sig, 'sum(iterable, /, start)')
        self.assertEqual(signa, 'sum()')

        sig, signa = _get_signatures('sum', sum, name_no_args='nna', max_args=99)
        self.assertEqual(sig, 'sum(iterable, /, start)')
        self.assertEqual(signa, 'nna()')

    def test_interface_get_signatures_d(self) -> None:

        sig, signa = _get_signatures('sum', sum, delegate_func=sum, delegate_name='sum2', max_args=99)
        self.assertEqual(sig, 'sum(iterable, /, start).sum2(iterable, /, start)')
        self.assertEqual(signa, 'sum().sum2()')

    def test_interface_get_signatures_e(self) -> None:

        sig, signa = _get_signatures('sum', sum, delegate_func=sum, delegate_name='sum2', terminus_func=sum, terminus_name='sum3', max_args=99)
        self.assertEqual(sig, 'sum(iterable, /, start).sum2(iterable, /, start).sum3(iterable, /, start)')
        self.assertEqual(signa, 'sum().sum2().sum3()')

    def test_interface_get_signatures_f(self) -> None:

        sig, signa = _get_signatures('sum', sum, delegate_func=sum, delegate_name='sum2', delegate_namespace='dns', terminus_func=sum, terminus_name='sum3', max_args=99)
        self.assertEqual(sig, 'sum(iterable, /, start).dns.sum2(iterable, /, start).sum3(iterable, /, start)')
        self.assertEqual(signa, 'sum().dns.sum2().sum3()')

    def test_interface_get_signatures_g(self) -> None:
        # if we provide a func and no name, we assume that func is a __call__ on the parent object
        sig, signa = _get_signatures('sum', sum, delegate_func=sum, delegate_name='sum2', terminus_func=min, max_args=99)
        self.assertEqual(sig, 'sum(iterable, /, start).sum2(iterable, /, start)()')
        self.assertEqual(signa, 'sum().sum2()()')

    def test_interface_get_signatures_h(self) -> None:
        # if we provide a func and no name, we assume that func is a __call__ on the parent object
        sig, signa = _get_signatures('sum', sum, delegate_func=sum, terminus_func=min, max_args=99)
        self.assertEqual(sig, 'sum(iterable, /, start)(iterable, /, start)()')
        self.assertEqual(signa, 'sum()()()')

    #---------------------------------------------------------------------------

    def test_interface_get_frame_a(self) -> None:

        f1 = InterfaceSummary.to_frame(Series)
        f2 = InterfaceSummary.to_frame(Series, minimized=False, max_args=np.inf)
        self.assertTrue(len(f1) == len(f2))

        self.assertEqual(f1.columns.values.tolist(),
                ['cls_name', 'group', 'doc']
                )
        self.assertEqual(
                f2.columns.values.tolist(),
                ['cls_name', 'group', 'doc', 'reference', 'use_signature', 'is_attr', 'delegate_reference', 'delegate_is_attr', 'signature_no_args', 'sna_label']
                )

    def test_interface_assign_a(self) -> None:
        f = Frame.interface.loc[Frame.interface.index.via_str.startswith('assign')]
        # assignment interface is one of the most complex, so we can check the signatures here explicitly
        self.assertEqual(f.index.values.tolist(),
                ['assign[key](value, *, fill_value)', 'assign[key].apply(func, *, fill_value)', 'assign[key].apply_element(func, *, dtype, fill_value)', 'assign[key].apply_element_items(func, *, dtype, fill_value)', 'assign.iloc[key](value, *, fill_value)', 'assign.iloc[key].apply(func, *, fill_value)', 'assign.iloc[key].apply_element(func, *, dtype, fill_value)', 'assign.iloc[key].apply_element_items(func, *, dtype, fill_value)', 'assign.loc[key](value, *, fill_value)', 'assign.loc[key].apply(func, *, fill_value)', 'assign.loc[key].apply_element(func, *, dtype, fill_value)', 'assign.loc[key].apply_element_items(func, *, dtype, fill_value)', 'assign.bloc[key](value, *, fill_value)', 'assign.bloc[key].apply(func, *, fill_value)', 'assign.bloc[key].apply_element(func, *, dtype, fill_value)', 'assign.bloc[key].apply_element_items(func, *, dtype, fill_value)'])

    #---------------------------------------------------------------------------

    def test_interface_via_re_signature_no_args(self) -> None:
        inter = InterfaceSummary.to_frame(Series,
                minimized=False,
                max_args=99, # +inf, but keep as int
                )
        self.assertEqual(
            inter.loc[inter['group']==InterfaceGroup.AccessorFillValue, 'signature_no_args'].values.tolist(),
            ['via_fill_value().loc', 'via_fill_value().__getitem__()', 'via_fill_value().via_T', 'via_fill_value().__add__()', 'via_fill_value().__sub__()', 'via_fill_value().__mul__()', 'via_fill_value().__truediv__()', 'via_fill_value().__floordiv__()', 'via_fill_value().__mod__()', 'via_fill_value().__pow__()', 'via_fill_value().__lshift__()', 'via_fill_value().__rshift__()', 'via_fill_value().__and__()', 'via_fill_value().__xor__()', 'via_fill_value().__or__()', 'via_fill_value().__lt__()', 'via_fill_value().__le__()', 'via_fill_value().__eq__()', 'via_fill_value().__ne__()', 'via_fill_value().__gt__()', 'via_fill_value().__ge__()', 'via_fill_value().__radd__()', 'via_fill_value().__rsub__()', 'via_fill_value().__rmul__()', 'via_fill_value().__rtruediv__()', 'via_fill_value().__rfloordiv__()']
            )

        self.assertEqual(
            inter.loc[inter['group']==InterfaceGroup.AccessorRe, 'signature_no_args'].values.tolist(),
            ['via_re().search()', 'via_re().match()', 'via_re().fullmatch()', 'via_re().split()', 'via_re().findall()', 'via_re().sub()', 'via_re().subn()'])

    #---------------------------------------------------------------------------

    def test_interface_get_instance(self) -> None:
        for component in DOCUMENTED_COMPONENTS:
            post = InterfaceSummary.get_instance(component)
            self.assertTrue(post is not None)

    #---------------------------------------------------------------------------

    def test_interface_util_a(self) -> None:
        f = InterfaceSummary.to_frame(FillValueAuto, minimized=False, max_args=99)
        self.assertTrue(f.size > 0)
        self.assertEqual(len(f), len(f['sna_label'].unique()))


    def test_interface_summary_name_obj_iter_a(self) -> None:
        inter = InterfaceSummary.to_frame(WWW,
                minimized=False,
                max_args=99,
                )
        self.assertEqual(inter['signature_no_args'].values.tolist(),
            ['from_file()', 'from_gzip()', 'from_zip()']
            )

    def test_interface_summary_name_obj_iter_b(self) -> None:
        inter = InterfaceSummary.to_frame(Require,
                minimized=False,
                max_args=99,
                )
        sigs = inter['signature_no_args'].values.tolist()
        assert 'Len()' in sigs
        assert 'Name()' in sigs

    #---------------------------------------------------------------------------
    def test_valid_argument_types_a(self) -> None:

        def a1(a, b, /, *, c, d): pass
        valid_argument_types(a1)

        def a2(a, b, /): pass
        valid_argument_types(a2)

        def a3(): pass
        valid_argument_types(a3)

        def a4(*, c, d): pass
        valid_argument_types(a4)

        def b(a, b, *, c, d): pass
        with self.assertRaises(RuntimeError):
            valid_argument_types(b)

        def c(a, *, c, d): pass
        with self.assertRaises(RuntimeError):
            valid_argument_types(c)

        def d(a): pass
        with self.assertRaises(RuntimeError):
            valid_argument_types(d)


    def test_valid_argument_types_b(self) -> None:

        class A:
            def __init__(self): pass
            def a(self, a, b, /): pass
            def b(self, a, b): pass

            @classmethod
            def c(cls, a, b, /): pass

            @classmethod
            def d(cls, a): pass

            @classmethod
            def e(cls, self): pass

            @staticmethod
            def f(self, a, /): pass # noqa: PLW0211

            @staticmethod
            def g(self, a, b, /): pass # noqa: PLW0211


        valid_argument_types(A.__init__)

        valid_argument_types(A.a)

        valid_argument_types(A.c)

        with self.assertRaises(RuntimeError):
            valid_argument_types(A.b)

        with self.assertRaises(RuntimeError):
            valid_argument_types(A.d)

        with self.assertRaises(RuntimeError):
            valid_argument_types(A.e)

        valid_argument_types(A.f)

        # note: we cannot distinguish betweeen instance method and static method
        valid_argument_types(A.g)


    def test_interfaces(self) -> None:

        exclude = {'__array__',
                '__array_ufunc__',
                '__deepcopy__',
                '__setstate__',
                '__getitem__',
                '__setitem__',
                '__dataframe__',
                'relabel',
                'relabel_flat',
                'relabel_level_add',
                'relabel_level_drop',
                'reindex',
                'roll',
                'shift',
                'sample',
                'pivot',
                'rehierarch',
                'get',
                'difference', # *args
                'intersection', # *args
                'union', # *args
                'from_date_range',
                'from_year_month_range',
                'from_year_range',
                   }

        for target in (
                Series,
                Frame,
                FrameGO,
                Index,
                IndexDate,
                IndexDateGO,
                IndexHierarchy,
                Bus,
                Batch,
                Yarn,
                Quilt,
                ):
            for name_attr, obj, _ in InterfaceSummary.name_obj_iter(target):
                if callable(obj) and name_attr not in exclude:
                    valid_argument_types(obj)



if __name__ == '__main__':
    import unittest
    unittest.main()
