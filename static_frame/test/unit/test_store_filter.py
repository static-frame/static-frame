import unittest
# from io import StringIO
import numpy as np # type: ignore



from static_frame.core.store_filter import STORE_FILTER_DEFAULT
from static_frame.test.test_case import TestCase
# from static_frame.test.test_case import temp_file

class TestUnit(TestCase):

    def test_store_from_type_filter_array_a(self) -> None:

        a1 = np.array([1, 2, np.nan, np.inf, -np.inf], dtype=float)
        a2 = np.array([1, 2, np.nan, np.inf, -np.inf], dtype=object)

        sfd = STORE_FILTER_DEFAULT

        self.assertEqual(sfd.from_type_filter_array(a1).tolist(),
                [1.0, 2.0, '', '+inf', '-inf'])

        self.assertEqual(sfd.from_type_filter_array(a2).tolist(),
                [1.0, 2.0, '', '+inf', '-inf'])

    def test_store_from_type_filter_array_b(self) -> None:

        a1 = np.array([False, 20, False], dtype=bool)
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
            [1, 'None', '', '-inf', '+inf'])


    def test_store_from_type_filter_element_a(self) -> None:
        sfd = STORE_FILTER_DEFAULT

        self.assertEqual(sfd.from_type_filter_element(None), 'None')
        self.assertEqual(sfd.from_type_filter_element(np.nan), '')




if __name__ == '__main__':
    unittest.main()
