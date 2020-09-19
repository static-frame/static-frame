import unittest
from io import StringIO
import datetime

import numpy as np

from static_frame.core.store_filter import STORE_FILTER_DEFAULT
from static_frame.core.store_filter import STORE_FILTER_DISABLE
from static_frame.core.store_filter import StoreFilter
from static_frame.test.test_case import TestCase
from static_frame.core.util import DTYPE_OBJECT_KIND
from static_frame.core.util import DTYPE_COMPLEX_KIND
from static_frame.core.util import DTYPE_FLOAT_KIND
from static_frame.core.util import NAT


from static_frame.core.frame import Frame


class TestUnit(TestCase):

    def test_store_from_type_filter_array_a(self) -> None:

        a1 = np.array([1, 2, np.nan, np.inf, -np.inf], dtype=float)
        a2 = np.array([1, 2, np.nan, np.inf, -np.inf], dtype=object)

        sfd = STORE_FILTER_DEFAULT

        self.assertEqual(sfd.from_type_filter_array(a1).tolist(),
                [1.0, 2.0, '', 'inf', '-inf'])

        self.assertEqual(sfd.from_type_filter_array(a2).tolist(),
                [1.0, 2.0, '', 'inf', '-inf'])

    def test_store_from_type_filter_array_b(self) -> None:

        a1 = np.array([False, True, False], dtype=bool)
        a2 = np.array([False, True, False], dtype=object)

        sfd = STORE_FILTER_DEFAULT

        self.assertEqual(sfd.from_type_filter_array(a1).tolist(),
                [False, True, False])
        self.assertEqual(sfd.from_type_filter_array(a2).tolist(),
                [False, True, False])


    def test_store_from_type_filter_array_c(self) -> None:

        a1 = np.array([1, 20, 1], dtype=int)
        a2 = np.array([1, 20, 1], dtype=object)

        sfd = STORE_FILTER_DEFAULT

        self.assertEqual(sfd.from_type_filter_array(a1).tolist(),
                [1, 20, 1])
        self.assertEqual(sfd.from_type_filter_array(a2).tolist(),
                [1, 20, 1])


    def test_store_from_type_filter_array_d(self) -> None:

        a1 = np.array([1, None, np.nan, -np.inf, np.inf], dtype=object)

        sfd = STORE_FILTER_DEFAULT

        self.assertEqual(sfd.from_type_filter_array(a1).tolist(),
            [1, 'None', '', '-inf', 'inf'])


    def test_store_from_type_filter_array_e(self) -> None:

        a1 = np.array([1, None, np.nan, -np.inf, np.inf], dtype=object)

        sfd = STORE_FILTER_DISABLE

        self.assertAlmostEqualValues(
            sfd.from_type_filter_array(a1).tolist(),
            [1, None, np.nan, -np.inf, np.inf])

    def test_store_from_type_filter_array_f(self) -> None:

        a1 = np.array(['2012', '2013'], dtype=np.datetime64)

        sfd = STORE_FILTER_DEFAULT

        a2 = sfd.from_type_filter_array(a1)
        self.assertEqual(a2.tolist(),
                ['2012', '2013'])

        a3 = np.array(['2012', '2013', NAT], dtype=np.datetime64)
        a4 = sfd.from_type_filter_array(a3)
        self.assertEqual(a4.tolist(),
                ['2012', '2013', ''])


    def test_store_from_type_filter_array_g(self) -> None:

        sfd = STORE_FILTER_DEFAULT
        a1 = np.array(['2012-02-05', '2013-05-21', NAT], dtype=np.datetime64)
        a2 = sfd.from_type_filter_array(a1)
        self.assertEqual(a2.tolist(),
                [datetime.date(2012, 2, 5), datetime.date(2013, 5, 21), ''])

    #---------------------------------------------------------------------------
    def test_store_from_type_filter_element_a(self) -> None:
        sfd = STORE_FILTER_DEFAULT

        self.assertEqual(sfd.from_type_filter_element(None), 'None')
        self.assertEqual(sfd.from_type_filter_element(np.nan), '')


    def test_store_from_type_filter_element_b(self) -> None:
        sfd = STORE_FILTER_DEFAULT

        self.assertEqual(sfd.from_type_filter_element(np.nan), '')
        self.assertEqual(sfd.from_type_filter_element(np.nan-3j), '')

    def test_store_from_type_filter_element_c(self) -> None:
        sf1 = StoreFilter(
                value_format_float_positional='{:.2e}',
                value_format_float_scientific='{:.2e}')

        self.assertEqual(sf1.from_type_filter_element(3.1), '3.10e+00')
        self.assertEqual(sf1.from_type_filter_element(0.0000011193), '1.12e-06')



    #---------------------------------------------------------------------------

    def test_store_to_type_filter_element_a(self) -> None:
        sfd = STORE_FILTER_DEFAULT

        self.assertTrue(np.isnan(sfd.to_type_filter_element('nan')))
        self.assertTrue(np.isposinf(sfd.to_type_filter_element('inf')))
        self.assertTrue(np.isneginf(sfd.to_type_filter_element('-inf')))
        self.assertEqual(sfd.to_type_filter_element('None'), None)

    def test_store_to_type_filter_element_b(self) -> None:
        sfd = STORE_FILTER_DISABLE

        self.assertEqual(sfd.to_type_filter_element('nan'), 'nan')
        self.assertEqual(sfd.to_type_filter_element('inf'), 'inf')
        self.assertEqual(sfd.to_type_filter_element('-inf'), '-inf')
        self.assertEqual(sfd.to_type_filter_element('None'), 'None')


    #---------------------------------------------------------------------------

    def test_store_to_type_filter_array_a(self) -> None:
        sfd = STORE_FILTER_DEFAULT
        a1 = np.array([1, None, 'nan', '', 'inf'], dtype=object)
        post = sfd.to_type_filter_array(a1)
        self.assertAlmostEqualValues(post.tolist(), [1, None, np.nan, np.nan, np.inf])


    def test_store_to_type_filter_array_b(self) -> None:
        sfd = STORE_FILTER_DEFAULT
        a1 = np.array(['2012', '2013'], dtype=np.datetime64)
        post = sfd.to_type_filter_array(a1)
        self.assertAlmostEqualValues(post.tolist(), [datetime.date(2012, 1, 1), datetime.date(2013, 1, 1)])


    #---------------------------------------------------------------------------

    def test_store_filter_to_delimited_a(self) -> None:
        f = Frame.from_records(((None, np.inf), (np.nan, -np.inf)))
        store_filter = StoreFilter(from_nan='*', from_none='!', from_posinf='&', from_neginf='@')
        post = StringIO()
        f.to_csv(post, store_filter=store_filter, include_index=False)
        post.seek(0)
        self.assertEqual(post.read(), '0,1\n!,&\n*,@\n')


    #---------------------------------------------------------------------------

    def test_store_filter_format_inexact_element_a(self) -> None:
        sf1 = StoreFilter(
                value_format_float_positional='{:.2e}',
                value_format_float_scientific='{:.2e}')

        self.assertEqual(
                sf1._format_inexact_element(0.000000020, DTYPE_FLOAT_KIND),
                '2.00e-08'
                )
        self.assertEqual(
                sf1._format_inexact_element(0.000001119, DTYPE_OBJECT_KIND),
                '1.12e-06'
                )

        self.assertEqual(
                sf1._format_inexact_element(20, DTYPE_OBJECT_KIND),
                20
                )

    def test_store_filter_format_inexact_element_b(self) -> None:
        sf1 = StoreFilter(
                value_format_float_positional='{:.4f}',
                value_format_float_scientific='{:.4e}')

        self.assertEqual(
                sf1._format_inexact_element(0.832555, DTYPE_FLOAT_KIND),
                '0.8326'
                )
        self.assertEqual(
                sf1._format_inexact_element(0.0000011193, DTYPE_OBJECT_KIND),
                '1.1193e-06'
                )
        self.assertEqual(
                sf1._format_inexact_element('foo', DTYPE_OBJECT_KIND),
                'foo'
                )

    def test_store_filter_format_inexact_element_c(self) -> None:
        sf1 = StoreFilter(
                value_format_float_positional='{:.8f}',
                value_format_float_scientific='{:.8f}')

        self.assertEqual(
                sf1._format_inexact_element(0.83255500, DTYPE_FLOAT_KIND),
                '0.83255500'
                )
        self.assertEqual(
                sf1._format_inexact_element(0.0000011193, DTYPE_OBJECT_KIND),
                '0.00000112'
                )
        self.assertEqual(
                sf1._format_inexact_element(False, DTYPE_OBJECT_KIND),
                False
                )

    def test_store_filter_format_inexact_element_d(self) -> None:
        sf1 = StoreFilter(
                value_format_complex_positional='{:.2e}',
                value_format_complex_scientific='{:.2e}')

        self.assertEqual(
                sf1._format_inexact_element(0.000001-0.0000005j, DTYPE_COMPLEX_KIND),
                '1.00e-06-5.00e-07j'
                )
        self.assertEqual(
                sf1._format_inexact_element(20+3j, DTYPE_OBJECT_KIND),
                '2.00e+01+3.00e+00j'
                )

    def test_store_filter_format_inexact_element_e(self) -> None:
        sf1 = StoreFilter(
                value_format_complex_positional='{:.8f}',
                value_format_complex_scientific='{:.4e}')

        self.assertEqual(
                sf1._format_inexact_element(0.4123-0.593j, DTYPE_COMPLEX_KIND),
                '0.41230000-0.59300000j'
                )
        # if either part goes to scientific, scientific is used on both parts
        self.assertEqual(
                sf1._format_inexact_element(0.413-0.000000593j, DTYPE_OBJECT_KIND),
                '4.1300e-01-5.9300e-07j'
                )


    #---------------------------------------------------------------------------
    def test_store_filter_format_inexact_array_a(self) -> None:
        sf1 = StoreFilter(
                value_format_float_positional='{:.8f}',
                value_format_float_scientific='{:.8f}',
                value_format_complex_positional='{:.8f}',
                value_format_complex_scientific='{:.8f}',
                )

        a1 = np.array([0.83255500, 0.0000011193, 0.832555, 20])
        a2 = np.array([0.413-0.000000593j, 0.4123-0.593j, 0.832555, 20+3j])

        post1 = sf1._format_inexact_array(a1, None)
        self.assertEqual(post1.tolist(), ['0.83255500', '0.00000112', '0.83255500', '20.00000000'])

        post2 = a2.astype(object)
        post3 = sf1._format_inexact_array(a2, post2)
        self.assertEqual(id(post3), id(post2))

        self.assertEqual(post3.tolist(),
                ['0.41300000-0.00000059j', '0.41230000-0.59300000j', '0.83255500+0.00000000j', '20.00000000+3.00000000j'])

        # originate with object arrays
        a3 = np.array([0.83255500, 0.0000011193, 0.832555, 20]).astype(object)
        a4 = np.array([0.413-0.000000593j, 0.4123-0.593j, 0.832555, 20+3j]).astype(object)

        post4 = sf1._format_inexact_array(a3, None)
        self.assertEqual(post4.tolist(), ['0.83255500', '0.00000112', '0.83255500', '20.00000000'])

        post5 = a4.astype(object)
        post6 = sf1._format_inexact_array(a4, post5)
        self.assertEqual(id(post5), id(post6))

        self.assertEqual(post6.tolist(),
                ['0.41300000-0.00000059j', '0.41230000-0.59300000j', '0.83255500+0.00000000j', '20.00000000+3.00000000j'])


    def test_store_filter_format_inexact_array_b(self) -> None:
        sf1 = StoreFilter(
                value_format_float_positional='{:.2e}',
                value_format_float_scientific='{:.2e}',
                value_format_complex_positional='{:.2e}',
                value_format_complex_scientific='{:.2e}',
                )

        a1 = np.array([0.83255500, 0.0000011193, 0.832555, 20]).reshape(2, 2)
        a2 = np.array([0.413-0.000000593j, 0.4123-0.593j, 0.832555, 20+3j]).reshape(2, 2)

        post1 = sf1._format_inexact_array(a1, None)
        post2 = sf1._format_inexact_array(a2, None)

        self.assertEqual(post1.tolist(),
                [['8.33e-01', '1.12e-06'], ['8.33e-01', '2.00e+01']]
                )

        self.assertEqual(post2.tolist(),
                [['4.13e-01-5.93e-07j', '4.12e-01-5.93e-01j'],
                ['8.33e-01+0.00e+00j', '2.00e+01+3.00e+00j']])


    def test_store_filter_format_inexact_array_c(self) -> None:
        sf1 = StoreFilter(
                value_format_float_positional='{:.3f}',
                value_format_float_scientific='{:.3f}',
                value_format_complex_positional='{:.3f}',
                value_format_complex_scientific='{:.3f}',
                )

        a1 = np.array([0.83255500, None, 0.0000011193, 0.832555, 20, True], dtype=object)
        a2 = np.array([0.413-0.000000593j, 0.4123-0.593j, 'foo', False, 100, 0.832555, 20+3j], dtype=object)

        post1 = sf1._format_inexact_array(a1, None)
        post2 = sf1._format_inexact_array(a2, None)

        self.assertEqual(post1.tolist(),
                ['0.833', None, '0.000', '0.833', 20, True])

        self.assertEqual(post2.tolist(),
                ['0.413-0.000j', '0.412-0.593j', 'foo', False, 100, '0.833', '20.000+3.000j'])



if __name__ == '__main__':
    unittest.main()
