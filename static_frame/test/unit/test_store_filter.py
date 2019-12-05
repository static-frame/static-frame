import unittest
# from io import StringIO
import numpy as np
from io import StringIO

from static_frame.core.store_filter import STORE_FILTER_DEFAULT
from static_frame.core.store_filter import STORE_FILTER_DISABLE
from static_frame.core.store_filter import StoreFilter
from static_frame.test.test_case import TestCase

from static_frame.core.frame import Frame
# from static_frame.test.test_case import temp_file

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




    def test_store_from_type_filter_element_a(self) -> None:
        sfd = STORE_FILTER_DEFAULT

        self.assertEqual(sfd.from_type_filter_element(None), 'None')
        self.assertEqual(sfd.from_type_filter_element(np.nan), '')


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



    def test_store_to_type_filter_array_a(self) -> None:
        sfd = STORE_FILTER_DEFAULT
        a1 = np.array([1, None, 'nan', '', 'inf'], dtype=object)
        post = sfd.to_type_filter_array(a1)
        self.assertAlmostEqualValues(post.tolist(), [1, None, np.nan, np.nan, np.inf])



    def test_store_filter_to_delimited_a(self) -> None:
        f = Frame.from_records(((None, np.inf), (np.nan, -np.inf)))
        store_filter = StoreFilter(from_nan='*', from_none='!', from_posinf='&', from_neginf='@')
        post = StringIO()
        f.to_csv(post, store_filter=store_filter, include_index=False)
        post.seek(0)
        self.assertEqual(post.read(), '0,1\n!,&\n*,@')


if __name__ == '__main__':
    unittest.main()
